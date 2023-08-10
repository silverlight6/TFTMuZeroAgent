import ray
import config
import numpy as np
from Models.MuZero_torch_agent import MuZeroNetwork as TFTNetwork
from checkpoint import Checkpoint
from sys import getsizeof


@ray.remote(num_gpus=0.01)
class Storage:
    def __init__(self, episode):
        self.target_model = self.load_model()
        if episode > 0:
            self.target_model.tft_load_model(episode)
        self.model = self.target_model
        self.episode_played = 0
        self.placements = {"player_" + str(r): [0 for _ in range(config.NUM_PLAYERS)]
                           for r in range(config.NUM_PLAYERS)}
        self.trainer_busy = False
        self.checkpoint_list = np.array([], dtype=object)
        self.max_q_value = 1
        self.store_initial_checkpoint(episode)

    def get_model(self):
        return self.checkpoint_list[-1].get_model()

    # Implementing saving.
    def load_model(self):
        return TFTNetwork()

    def get_target_model(self):
        return self.target_model.get_weights()

    def set_target_model(self, weights):
        return self.target_model.set_weights(weights)

    def get_episode_played(self):
        return self.episode_played

    def increment_episode_played(self):
        self.episode_played += 1

    def set_trainer_busy(self, status):
        self.trainer_busy = status

    def get_trainer_busy(self):
        return self.trainer_busy

    def record_placements(self, placement):
        print(placement)
        for key in self.placements.keys():
            # Increment which position each model got.
            self.placements[key][placement[key]] += 1

    def store_initial_checkpoint(self, episode):
        base_checkpoint = Checkpoint(episode, 1)
        # TODO: Verify if this is super inefficient or if there is a better way.
        self.checkpoint_list = np.append(self.checkpoint_list, [base_checkpoint])

    def store_checkpoint(self, episode):
        print("STORING CHECKPOINT {}".format(episode))
        checkpoint = Checkpoint(episode, self.max_q_value)
        self.checkpoint_list = np.append(self.checkpoint_list, [checkpoint])

        # Update this later to delete the model with the lowest value.
        # Want something so it doesn't expand infinitely
        if len(self.checkpoint_list) > 1000:
            self.checkpoint_list = self.checkpoint_list[1:]

    def update_checkpoint_score(self, episode, prob):
        print("UPDATING CHECKPOINT SCORE OF EPISODE {}".format(episode))
        checkpoint = next((x for x in self.checkpoint_list if x.epoch == episode), None)
        if checkpoint:
            checkpoint.update_q_score(self.checkpoint_list[-1].epoch, prob)
        print("UPDATED CHECKPOINT SCORE OF EPISODE {} TO {}".format(episode, checkpoint.q_score))

    def sample_past_model(self):
        # List of probabilities for each past model
        probabilities = np.array([], dtype=np.float32)

        # List of checkpoint epochs, so we can load the right model
        checkpoints = np.array([], dtype=np.float32)

        # Populate the lists
        for checkpoint in self.checkpoint_list:
            probabilities = np.append(probabilities, [np.exp(checkpoint.q_score)])
            checkpoints = np.append(checkpoints, [checkpoint.epoch])

        # Normalize the probabilities to create a probability distribution
        probabilities = probabilities / np.linalg.norm(probabilities)

        # Pick the sample
        choice = np.random.choice(checkpoints, 1, probabilities)

        # Find the index, so we can return the probability as well in case we need to update the value
        index = np.where(checkpoints == choice)[0][0]

        # Return the model and the probability
        return self.checkpoint_list[index].get_model(), choice, probabilities[index]

    # Going to be adding a bunch of new methods here.
    # One that is going to be used to call when I save checkpoints
    # One to create a checkpoint on the start of training that I can call from init.
    # I also need a list to store probabilities and a method to return an agent depending on the sample.
    # Remember to do a check on the max_q_value after the agent update.
