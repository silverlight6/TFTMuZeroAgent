import ray
import config
import time
import numpy as np


# @ray.remote(, )
@ray.remote(memory=config.BATCH_SIZE * 10000, num_cpus=2, num_gpus=0.1)
class GlobalBuffer:
    def __init__(self, storage_ptr):
        self.gameplay_experiences = PriorityBuffer(10000)
        self.batch_size = config.BATCH_SIZE
        self.storage_ptr = storage_ptr
        self.average_position = PriorityBuffer(10000)
        self.current_batch = []
        self.batch_full = False
        self.ckpt_time = time.time_ns()

    # Might be a bug with the action_batch not always having correct dims
    def sample_batch(self):
        # Returns: a batch of gameplay experiences without regard to which agent.
        obs_tensor_batch, action_history_batch, target_value_batch, policy_mask_batch = [], [], [], []
        target_reward_batch, target_policy_batch, value_mask_batch, reward_mask_batch = [], [], [], []
        sample_set_batch, tier_batch, final_tier_batch, champion_batch, position_batch = [], [], [], [], []
        for batch_num in range(self.batch_size):
            # Setting the position gameplay_experiences get and position get next to each other to try to minimize
            # The number of multiprocessing errors that could occur by having them be too far apart.

            observation, action_history, value_mask, reward_mask, policy_mask, value, reward, policy, \
                sample_set, tier_set, final_tier_set, champion_set = self.gameplay_experiences.extractMax()
            position_batch.append(self.average_position.extractMax())
            obs_tensor_batch.append(observation)
            action_history_batch.append(action_history[1:])
            value_mask_batch.append(value_mask)
            reward_mask_batch.append(reward_mask)
            policy_mask_batch.append(policy_mask)
            target_value_batch.append(value)
            target_reward_batch.append(reward)
            target_policy_batch.append(policy)
            sample_set_batch.append(sample_set)
            tier_batch.append(tier_set)
            final_tier_batch.append(final_tier_set)
            champion_batch.append(champion_set)

        # observation_batch = np.squeeze(np.asarray(obs_tensor_batch))
        observation_batch = self.reshape_observation(obs_tensor_batch)
        action_history_batch = np.asarray(action_history_batch)
        target_value_batch = np.asarray(target_value_batch).astype('float32')
        target_reward_batch = np.asarray(target_reward_batch).astype('float32')
        value_mask_batch = np.asarray(value_mask_batch).astype('float32')
        reward_mask_batch = np.asarray(reward_mask_batch).astype('float32')
        policy_mask_batch = np.asarray(policy_mask_batch).astype('float32')
        position_batch = np.asarray(position_batch)
        position_batch = np.mean(position_batch)

        data_list = [
            observation_batch, action_history_batch, value_mask_batch, reward_mask_batch,
            policy_mask_batch, target_value_batch, target_reward_batch, target_policy_batch,
            sample_set_batch, tier_batch, final_tier_batch, champion_batch, np.array(position_batch)
        ]
        return np.array(data_list, dtype=object)

    def reshape_observation(self, obs_batch):
        obs_reshaped = {}

        for obs in obs_batch:
            for key in obs:
                if key not in obs_reshaped:
                    obs_reshaped[key] = []
                obs_reshaped[key].append(obs[key])

        for key in obs_reshaped:
            obs_reshaped[key] = np.stack(obs_reshaped[key], axis=0)

        return obs_reshaped

    def store_replay_sequence(self, samples):
        for sample in samples[0]:
            if self.gameplay_experiences.size / 25000 < 0.9 and sample[0] > 1:
                self.gameplay_experiences.insert(sample[0], sample[1])
                self.average_position.insert(sample[0], samples[1])

    def available_batch(self):
        queue_length = self.gameplay_experiences.size
        if queue_length >= self.batch_size and not ray.get(self.storage_ptr.get_trainer_busy.remote()):
            print("QUEUE_LENGTH {} at time {}".format(queue_length, time.time_ns()))
            self.storage_ptr.set_trainer_busy.remote(True)
            return True
        time.sleep(1)
        print("QUEUE_LENGTH_SLEEPY {} at time {}".format(queue_length, time.time_ns()))
        return False

    # Leaving this transpose method here in case some model other than
    # MuZero requires this in the future.
    def transpose(self, matrix):
        rows = len(matrix)
        columns = len(matrix[0])

        matrix_T = []
        for j in range(columns):
            row = []
            for i in range(rows):
                row.append(matrix[i][j])
            matrix_T.append(row)

        return matrix_T


# Code taken from https://www.geeksforgeeks.org/max-heap-in-python/ and adjusted for the use-case
class PriorityBuffer:
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.size = 0
        self.Heap = [0] * (self.maxsize + 1)
        self.Replay_Heap = [None] * (self.maxsize + 1)
        self.Heap[0] = 0
        self.FRONT = 1

    # Function to return the position of
    # parent for the node currently
    # at pos
    def parent(self, pos):
        return pos // 2

    # Function to return the position of
    # the left child for the node currently
    # at pos
    def leftChild(self, pos):
        return 2 * pos

    # Function to return the position of
    # the right child for the node currently
    # at pos
    def rightChild(self, pos):
        return (2 * pos) + 1

    # Function that returns true if the passed
    # node is a leaf node
    def isLeaf(self, pos):
        if (self.size // 2) <= pos <= self.size:
            return True
        return False

    # Function to swap two nodes of the heap
    def swap(self, fpos, spos):
        self.Heap[fpos], self.Heap[spos] = (self.Heap[spos], self.Heap[fpos])
        self.Replay_Heap[fpos], self.Replay_Heap[spos] = (self.Replay_Heap[spos], self.Replay_Heap[fpos])

    # Function to heapify the node at pos
    def maxHeapify(self, pos):
        # If the node is a non-leaf node and smaller
        # than any of its child
        if not self.isLeaf(pos):
            if (self.Heap[pos] < self.Heap[self.leftChild(pos)] or
                    self.Heap[pos] < self.Heap[self.rightChild(pos)]):

                # Swap with the left child and heapify
                # the left child
                if (self.Heap[self.leftChild(pos)] >
                        self.Heap[self.rightChild(pos)]):
                    self.swap(pos, self.leftChild(pos))
                    self.maxHeapify(self.leftChild(pos))

                # Swap with the right child and heapify
                # the right child
                else:
                    self.swap(pos, self.rightChild(pos))
                    self.maxHeapify(self.rightChild(pos))

    # Function to insert a node into the heap
    def insert(self, priority, time_step):
        if self.size >= self.maxsize:
            # Reimplement this to instead of remove the last and insert at a random location.
            return
        self.size += 1
        self.Heap[self.size] = priority
        self.Replay_Heap[self.size] = time_step

        current = self.size

        while (self.Heap[current] >
               self.Heap[self.parent(current)]):
            self.swap(current, self.parent(current))
            current = self.parent(current)

    # Function to print the contents of the heap
    def Print(self):
        for i in range(1, (self.size // 2) + 1):
            print("PARENT : " + str(self.Heap[i]) +
                  "LEFT CHILD : " + str(self.Heap[2 * i]) +
                  "RIGHT CHILD : " + str(self.Heap[2 * i + 1]))

    # Function to remove and return the maximum
    # element from the heap
    def extractMax(self):
        popped = self.Replay_Heap[self.FRONT]
        self.Heap[self.FRONT] = self.Heap[self.size]
        self.Replay_Heap[self.FRONT] = self.Replay_Heap[self.size]
        self.size -= 1
        self.maxHeapify(self.FRONT)

        return popped
