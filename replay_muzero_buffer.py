import numpy as np
import config


class ReplayBuffer:
    def __init__(self, g_buffer):
        self.gameplay_experiences = []
        self.observation_history = []
        self.g_buffer = g_buffer

    def store_replay_buffer(self, observation, history, value, reward, policy, done):
        # Records a single step of gameplay experience
        # First few are self-explanatory
        # done is boolean if game is done after taking said action
        reward = np.clip(reward, -1.0, 1.0)
        self.gameplay_experiences.append([observation, history, value, reward, policy])
        frame_index = len(self.gameplay_experiences)

        if (frame_index % config.BATCH_SIZE == 0 and frame_index > 0) or (done and frame_index > config.BATCH_SIZE):
            # first create the batch
            observation_batch, history_batch, value_batch, reward_batch,  policy_batch = [], [], [], [], []
            for gameplay_experience in list(self.gameplay_experiences)[-16:]:
                observation_batch.append(gameplay_experience[0])
                history_batch.append(gameplay_experience[1])
                value_batch.append(gameplay_experience[2])
                reward_batch.append(gameplay_experience[3])
                policy_batch.append(gameplay_experience[4])
            self.g_buffer.store_replay_sequence(observation_batch, history_batch, value_batch, reward_batch,
                                                policy_batch)

    def store_observation(self, observation):
        self.observation_history.append(observation)

    def len_observation_buffer(self):
        return len(self.observation_history)

    def get_prev_observation(self, i):
        # take the sample from i num from the end of list
        return self.observation_history[i * -1]

    def get_observation_shape(self):
        # Hard coding this because the need to know this value before any observation are
        # Generated in the case of no successful actions completed yet in the game which
        # Is very likely at the start of the game.
        return config.OBSERVATION_SHAPE
