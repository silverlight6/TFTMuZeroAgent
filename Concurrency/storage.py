import config
import glob
import numpy as np
import ray
from Concurrency.checkpoint import Checkpoint
from config import STORAGE_GPU_SIZE
from Core.TorchModels.Muzero_default_agent import MuZeroDefaultNetwork as DefaultNetwork
from Core.TorchModels.MuZero_torch_agent import MuZeroNetwork as TFTNetwork
from Core.TorchModels.MuZero_position_torch_agent import MuZero_Position_Network as PositionNetwork
from Core.TorchModels.Representations.rep_testing_model import RepresentationTesting as RepNetwork
from Evaluator.eval_visualizers import GameResultPlotter


@ray.remote(num_gpus=STORAGE_GPU_SIZE, num_cpus=0.1)
class Storage:
    """
    Class that stores the global agent and other meta data that all of the data workers can access.
    Stores all checkpoints. Also stores a boolean to know if the trainer is currently busy or not.

    Args:
        episode (int): Checkpoint number to load in for the global agent.
    """
    def __init__(self, episode):
        self.target_model = self.load_model()
        if episode > 0:
            optimizer_dict = self.target_model.tft_load_model(episode)
        else:
            optimizer_dict = None
        self.model = self.target_model
        self.episode_played = 0
        # This is used for standard games
        self.trainer_busy = False
        self.checkpoint_list = np.array([], dtype=object)
        self.max_q_value = 1
        self.store_base_checkpoint()
        self.populate_checkpoints()
        self.placements = {"player_" + str(r): [0 for _ in range(config.NUM_PLAYERS)]
                           for r in range(config.NUM_PLAYERS)}
        # This is for single player games
        self.single_player_placements = {}
        # This is for placement
        self.reward_buffer = []
        self.optimizer_dict = optimizer_dict

        self.plotter = GameResultPlotter.remote()

    def get_model(self):
        """
        Returns the most recent checkpoint.

        Returns:
            Pytorch Model Weights:
                Model related to the most recent checkpoint
        """
        return self.checkpoint_list[-1].get_model()

    def load_model(self):
        """
        Returns a new model.

        Returns:
            Pytorch Model Weights:
                Weights for a fresh model
        """
        if config.CHAMP_DECIDER:
            return DefaultNetwork(config.ModelConfig())
        elif config.REP_TRAINER:
            return RepNetwork(config.ModelConfig())
        elif config.MUZERO_POSITION:
            return PositionNetwork(config.ModelConfig())
        else:
            return TFTNetwork(config.ModelConfig())

    def get_target_model(self):
        """
        Returns the current target model weights.

        Returns:
            Pytorch Model Weights:
                Target model weights.
        """
        return self.target_model.get_weights()

    def set_target_model(self, state_dict):
        """
        Sets new weights for the target model. Called after every gradiant update round.

        Args:
            Pytorch Model Weights:
                Weights of the target model.
        """
        return self.target_model.load_state_dict(state_dict)

    def save_target_model(self, train_step, optimizer):
        self.target_model.tft_save_model(train_step, optimizer)

    """
    Description - 
        Returns the current episode / train_step.
    Outputs     - 
        Current episode
    """
    def get_episode_played(self):
        return self.episode_played

    """
    Description - 
        Increments the current episode. Stored here so all data workers can access this.
    """
    def increment_episode_played(self):
        self.episode_played += 1

    """
    Description - 
        Updates the trainer_busy status. Called by available_batch and training loop after training.
    Inputs      - 
        status - boolean
            New status
    """
    def set_trainer_busy(self, status):
        self.trainer_busy = status

    """
    Description - 
        Returns the trainer_busy status. Called by available_batch.
    Outputs     - 
        current status
    """
    def get_trainer_busy(self):
        return self.trainer_busy

    def get_optimizer_dict(self):
        return self.optimizer_dict

    def get_checkpoint_list(self):
        return self.checkpoint_list

    """
    Description - 
    Inputs      - 
        placement - Dictionary
            placement of the agents. Checkpoint_num: position
    """
    def record_placements(self, placement):
        for key in self.placements.keys():
            # Increment which position each model got.
            self.placements[key][placement[key]] += 1

    """
    Description - 
        Inputs the checkpoint related to a fresh model into the checkpoint list.
    """
    def store_base_checkpoint(self):
        base_checkpoint = Checkpoint(0, 1, config.ModelConfig())
        # TODO: Verify if this is super inefficient or if there is a better way.
        self.checkpoint_list = np.append(self.checkpoint_list, [base_checkpoint])

    # TODO: Add description / inputs when doing unit testing on this method
    """
    Description -
    Inputs      - 
    """
    def store_checkpoint(self, episode):
        for checkpoint in self.checkpoint_list:
            if checkpoint.epoch == episode:
                return
        checkpoint = Checkpoint(episode, self.max_q_value, config.ModelConfig())
        self.checkpoint_list = np.append(self.checkpoint_list, [checkpoint])

        # Update this later to delete the model with the lowest value.
        # Want something so it doesn't expand infinitely
        if len(self.checkpoint_list) > 1000:
            self.checkpoint_list = self.checkpoint_list[1:]

    # TODO: Add description / inputs when doing unit testing on this method
    """
    Description - 
    Inputs      - 
    """
    def update_checkpoint_score(self, episode, prob):
        checkpoint = next((x for x in self.checkpoint_list if x.epoch == episode), None)
        if checkpoint:
            checkpoint.update_q_score(self.checkpoint_list[-1].epoch, prob)

    # TODO: Add description / outputs when doing unit testing on this method
    """
    Description - 
    Outputs     - 
    """
    def sample_past_model(self):
        # List of probabilities for each past model
        probabilities = np.array([], dtype=np.float32)

        # List of checkpoint epochs, so we can load the right model
        checkpoints = np.array([], dtype=np.float32)

        # Populate the lists
        for checkpoint in self.checkpoint_list:
            probabilities = np.append(probabilities, [np.exp(checkpoint.q_score)])
            checkpoints = np.append(checkpoints, [str(int(checkpoint.epoch))])

        # Normalize the probabilities to create a probability distribution
        probabilities = self.softmax(probabilities)

        # Pick the sample
        choice = np.random.choice(a=checkpoints, size=1, p=probabilities)

        # Find the index, so we can return the probability as well in case we need to update the value
        index = np.where(checkpoints == choice)[0][0]
        choice = int(choice)

        # Return the model and the probability
        return self.checkpoint_list[index].get_model(), choice, probabilities[index]

    # TODO: Add description when doing unit testing on this method
    """
    Description - 
    """
    # Method used to load in all checkpoints available on the system in the same folder as where we save checkpoints
    def populate_checkpoints(self):
        # Find all files within ./Checkpoints that follow the format Checkpoint_{integer}
        # Create a checkpoint and add it to the list for each occurrence it found.
        path_list = glob.glob('./Checkpoints/checkpoint_*')
        for path in path_list:
            checkpoint_int = int(path.split('_')[-1])
            self.store_checkpoint(checkpoint_int)

    # TODO: Add description / inputs when doing unit testing on this method
    """
    Description - 
    Outputs     - 
    """
    def get_checkpoint_list(self):
        return self.checkpoint_list

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    """
    Description - takes in an episode number and the recorded length of the episode, saves it and reports to 
                    tensorboard once there is enough data
    Inputs      - episode: int
                    what checkpoint the data is coming from
                  result: int
                    how far into the game that particular checkpoint made it at that time.
    """
    def record_game_result(self, episode, result):
        if str(episode) in self.single_player_placements.keys():
            self.single_player_placements[str(episode)].append(result)
        else:
            self.single_player_placements[str(episode)] = [result]
        if len(self.single_player_placements[str(episode)]) % 50 == 0:
            average = sum(self.single_player_placements[str(episode)]) / len(self.single_player_placements[str(episode)])
            max_distance = max(self.single_player_placements[str(episode)])
            min_distance = min(self.single_player_placements[str(episode)])
            result_data = {
                'average_game_length': average,
                'furthest_reached': max_distance,
                'worst_result': min_distance,
                'games_played': len(self.single_player_placements[str(episode)])
            }
            self.plotter.update_data.remote(episode, result_data)

    def reward_append(self, reward):
        self.reward_buffer.append(reward)
        if len(self.reward_buffer) > 500:
            self.reward_buffer = self.reward_buffer[200:]

    def return_reward(self):
        if len(self.reward_buffer) > 0:
            return sum(self.reward_buffer) / len(self.reward_buffer)



