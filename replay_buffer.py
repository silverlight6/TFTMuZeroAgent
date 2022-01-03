from collections import deque
import random
import numpy as np


class ReplayBuffer:
    def __init__(self):
        self.gameplay_experiences = deque(['state', 'action', 'reward', 'next_state', 'done'], maxlen=100000)

    def store_replay_buffer(self, state, action, reward, next_state, done):
        # Records a single step of gameplay experience
        # First few are self-explainatory
        # done is boolean if game is done after taking said action
        self.gameplay_experiences.append((state, action, reward, next_state, done))

    def sample_gameplay_batch(self):
        # samples a batch of gameplay experience for training. 
        # Returns: a list of gameplay experiences.
        batch_size = 256
        sampled_gameplay_batch = random.sample(self.gameplay_experiences, k=batch_size) \
            if len(self.gameplay_experiences) > batch_size else self.gameplay_experiences
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = [], [], [], [], [],
        # print(sampled_gameplay_batch[-1])
        for i in range(len(sampled_gameplay_batch) - 1, -1, -1):
            if len(sampled_gameplay_batch[i]) != 5:
                print("Found an error at i = " + str(i))
                sampled_gameplay_batch.remove(sampled_gameplay_batch[i])
                print("New length of gameplay_experiences is " + str(len(sampled_gameplay_batch)))
        for gameplay_experience in list(sampled_gameplay_batch):
            state_batch.append(gameplay_experience[0])
            action_batch.append(gameplay_experience[1])
            reward_batch.append(gameplay_experience[2])
            next_state_batch.append(gameplay_experience[3])
            done_batch.append(gameplay_experience[4])
        # print(len(state_batch))
        # print(len(state_batch[0][0]))
        return np.array(state_batch), np.array(action_batch), np.array(reward_batch), \
            np.array(next_state_batch), np.array(done_batch)
