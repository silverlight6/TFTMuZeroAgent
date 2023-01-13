import numpy as np
import config
import ray
from global_buffer import GlobalBuffer
from Models.replay_muzero_buffer import ReplayBuffer
import global_buffer as GlobalBuffer
from sklearn import preprocessing

@ray.remote
class BufferWrapper:
    def __init__(self, global_buffer):
        self.buffers = {"player_" + str(i) : ReplayBuffer(global_buffer) for i in range(config.NUM_PLAYERS)}
    
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

    def rewardNorm(self):
        reward_dat = []
        rewardLens = [0]

        for b in self.buffers.values():
            # clip rewards to prevent outliers from skewing results
            rewards = b.get_reward_sequence()
            print(rewards)
            rewards = np.clip(rewards, -3, 3)
            # store length of array to allocate elements later after normalization
            rewardLens.append(len(rewards))
            reward_dat.append(rewards)

        # reshape array of arrays of rewards to a single array
        # this reshaping should leave data from each reward array in order
        reward_dat = np.array(reward_dat)
        reward_dat.flatten()
        print(reward_dat[0])
        # normalize the values from this array w/ sklearn
        reward_dat = preprocessing.normalize(reward_dat)
        print(reward_dat[0])
        # reassign normalized values back into original arrays
        i = 0
        for b in self.buffers.values():
            print("setting")
            b.set_reward_sequence(rewards[rewardLens[i]:rewardLens[i+1]])
    
    def store_global_buffer(self):
        for b in self.buffers.values():
            b.store_global_buffer()
    