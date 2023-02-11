import numpy as np
import config
import ray
from Models.replay_muzero_buffer import ReplayBuffer as MuZeroBuffer
from Models.replay_a3c_buffer import ReplayBuffer as A3CBuffer
from sklearn import preprocessing


@ray.remote
class BufferWrapper:
    def __init__(self, global_buffer):
        if config.MODEL == "MuZero":
            self.buffers = {"player_" + str(i): MuZeroBuffer(global_buffer) for i in range(config.NUM_PLAYERS)}
        elif config.MODEL == "A3C":
            self.buffers = {"player_" + str(i): A3CBuffer(global_buffer) for i in range(config.NUM_PLAYERS)}
        else:
            self.buffers = {"player_" + str(i): MuZeroBuffer(global_buffer) for i in range(config.NUM_PLAYERS)}

    def store_replay_buffer(self, key, *args):
        self.buffers[key].store_replay_buffer(args[0], args[1], args[2], args[3])

    def store_observation(self, key, *args):
        self.buffers[key].store_observation(args[0])

    def len_observation_buffer(self, key):
        self.buffers[key].len_observation_buffer()

    def get_prev_observation(self, key):
        self.buffers[key].get_prev_observation()

    def get_prev_action(self, key):
        self.buffers[key].get_prev_action()

    def get_reward_sequence(self, key):
        self.buffers[key].get_reward_sequence()

    def set_reward_sequence(self, key, *args):
        self.buffers[key].set_reward_sequence(args[0])

    def store_global_buffer(self):
        for b in self.buffers.values():
            b.store_global_buffer()

    def sample_sequence(self, key, sequence_size):
        return self.buffers[key].sample_sequence(sequence_size)

    def sample_gameplay_batch(self, key):
        return self.buffers[key].sample_gameplay_batch()

    def rewardNorm(self):
        reward_dat = []
        rewardLens = []

        for b in self.buffers.values():
            # clip rewards to prevent outliers from skewing results
            rewards = b.get_reward_sequence()
            rewards = np.clip(rewards, -3, 3)
            # store length of array to allocate elements later after normalization
            rewardLens.append(len(rewards))
            reward_dat.append(rewards)

        # reshape array of arrays of rewards to a single array
        # this reshaping should leave data from each reward array in order
        reward_dat = np.array(reward_dat, dtype=object)
        reward_dat = np.hstack(reward_dat)
        # normalize the values from this array w/ sklearn
        reward_dat = preprocessing.scale(reward_dat)
        # reassign normalized values back into original arrays
        index = 0
        for i, b in enumerate(self.buffers.values()):
            b.set_reward_sequence(reward_dat[index: index + rewardLens[i]])
            index += rewardLens[i]


    