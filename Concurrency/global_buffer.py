import config
import time
import numpy as np
import asyncio
from Concurrency.priority_queue import PriorityBuffer

"""
Description - 
    Global Buffer that all of the data workers sent samples from completed games to.
    Uses a priority buffer. Also does all batch assembly for the trainer.
Inputs      - 
    storage_ptr
        pointer to the storage object to keep track of the current trainer status.
"""
class GlobalBuffer(object):
    def __init__(self, storage_ptr):
        self.gameplay_experiences = PriorityBuffer(10000)
        self.batch_size = config.BATCH_SIZE
        self.storage_ptr = storage_ptr
        self.average_position = PriorityBuffer(10000)
        self.current_batch = []
        self.batch_full = False
        self.ckpt_time = time.time_ns()

    # TODO: Add importance weights.
    """
    Description - 
        Prepares a batch for training. All preprocessing done here as to avoid problems with data transfer between cpu
        and gpu causing slow downs.
    Outputs     - 
        A prepared batch ready for training.
    """
    async def sample_batch(self):
        obs_tensor_batch, action_history_batch, target_value_batch, policy_mask_batch = [], [], [], []
        target_reward_batch, target_policy_batch, value_mask_batch, reward_mask_batch = [], [], [], []
        sample_set_batch, tier_batch, final_tier_batch, champion_batch, position_batch = [], [], [], [], []
        for batch_num in range(self.batch_size):
            # Setting the position gameplay_experiences get and position get next to each other to try to minimize
            # The number of multiprocessing errors that could occur by having them be too far apart.
            # If these two commands become off sync, it would cause the position logging to not match the rest
            # of the training logs but it would not break training.
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

    """
    Description - 
        Switches the batch and dictionary axis. 
        The model requires the dictionary to be in the first axis and the batch to be the second axis.
    """
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

    """
    Description - 
        Async method to store data into the global buffer. Some quick checking to ensure data validity.
    Inputs      - 
        samples
            All samples from one game from one agent.
    """
    async def store_replay_sequence(self, samples):
        for sample in samples[0]:
            if sample[0] > 1:
                self.gameplay_experiences.insert(sample[0], sample[1])
                self.average_position.insert(sample[0], samples[1])

    """
    Description - 
        Async method to determine if we have enough data in the buffer to start up the trainer.
    Outputs     - 
        True if there is data and trainer is free, false otherwise
    """
    async def available_batch(self):
        queue_length = self.gameplay_experiences.size
        if queue_length >= self.batch_size and not await self.storage_ptr.get_trainer_busy.remote():
            print("QUEUE_LENGTH {} at time {}".format(queue_length, time.time_ns()))
            await self.storage_ptr.set_trainer_busy.remote(True)
            return True
        await asyncio.sleep(2)
        # print("QUEUE_LENGTH_SLEEPY {} at time {}".format(queue_length, time.time_ns()))
        return False
