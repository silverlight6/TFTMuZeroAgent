import config
import time
import numpy as np
import asyncio
from Concurrency.priority_queue import PriorityBuffer


class GlobalBuffer(object):
    """
    Global Buffer that all of the data workers send samples from completed games to.
    Uses a priority buffer. Also does all batch assembly for the trainer.

    Args:
        storage_ptr (pointer): Pointer to the storage object to keep track of the current trainer status.
    """

    def __init__(self, storage_ptr):
        self.gameplay_experiences = PriorityBuffer(config.GLOBAL_BUFFER_SIZE)
        self.batch_size = config.BATCH_SIZE
        self.storage_ptr = storage_ptr
        self.average_position = PriorityBuffer(config.GLOBAL_BUFFER_SIZE)
        self.current_batch = []
        self.batch_full = False
        self.ckpt_time = time.time_ns()

    async def sample_batch(self):
        """
        Prepares a batch for training. All preprocessing done here to avoid problems with data transfer between CPU
        and GPU causing slowdowns.

        Conditions:
            - There is enough data in the buffer to train on.

        Returns:
            A prepared batch ready for training.
        """
        obs_tensor_batch, action_history_batch, target_value_batch, policy_mask_batch = [], [], [], []
        target_reward_batch, target_policy_batch, value_mask_batch, reward_mask_batch = [], [], [], []
        sample_set_batch, tier_batch, final_tier_batch, champion_batch, position_batch = [], [], [], [], []
        importance_weights = []
        for batch_num in range(self.batch_size):
            # Setting the position gameplay_experiences get and position get next to each other to minimize
            # the number of multiprocessing errors that could occur by having them too far apart.
            # If these two commands become out of sync, it would cause the position logging to not match the rest
            # of the training logs, but it would not break training.
            [observation, action_history, value_mask, reward_mask, policy_mask, value, reward, policy,
             sample_set, tier_set, final_tier_set, champion_set], priority = self.gameplay_experiences.extractMax()
            position, _ = self.average_position.extractMax()
            position_batch.append(position)
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
            importance_weights.append(1 / self.batch_size / priority)

        observation_batch = self.reshape_observation(obs_tensor_batch)
        action_history_batch = np.asarray(action_history_batch)
        target_value_batch = np.asarray(target_value_batch).astype('float32')
        target_reward_batch = np.asarray(target_reward_batch).astype('float32')
        value_mask_batch = np.asarray(value_mask_batch).astype('float32')
        reward_mask_batch = np.asarray(reward_mask_batch).astype('float32')
        policy_mask_batch = np.asarray(policy_mask_batch).astype('float32')
        importance_weights_batch = np.asarray(importance_weights).astype('float32')
        importance_weights_batch = importance_weights_batch / np.max(importance_weights_batch)
        position_batch = np.asarray(position_batch)
        position_batch = np.mean(position_batch)

        data_list = [
            observation_batch, action_history_batch, value_mask_batch, reward_mask_batch,
            policy_mask_batch, target_value_batch, target_reward_batch, target_policy_batch, sample_set_batch,
            importance_weights_batch, tier_batch, final_tier_batch, champion_batch, np.array(position_batch)
        ]
        return np.array(data_list, dtype=object)

    def reshape_observation(self, obs_batch):
        """
        Description:
            Switches the batch and dictionary axis.
            The model requires the dictionary to be in the first axis and the batch to be the second axis.
        """
        obs_reshaped = {}

        for obs in obs_batch:
            for key in obs:
                if key not in obs_reshaped:
                    obs_reshaped[key] = []
                obs_reshaped[key].append(obs[key])

        for key in obs_reshaped:
            obs_reshaped[key] = np.stack(obs_reshaped[key], axis=0)

        return obs_reshaped

    async def store_replay_sequence(self, samples):
        """
        Description:
            Async method to store data into the global buffer. Some quick checking to ensure data validity.

        Args:
            samples (list): All samples from one game from one agent.
        """
        for sample in samples[0]:
            if sample[0] > 1:
                self.gameplay_experiences.insert(sample[0], sample[1])
                self.average_position.insert(sample[0], samples[1])

    async def available_batch(self):
        """
        Description:
            Async method to determine if there is enough data in the buffer to start up the trainer.

        Outputs:
            - True if there is enough data and the trainer is free, false otherwise.
        """
        queue_length = self.gameplay_experiences.size
        if queue_length >= self.batch_size and not await self.storage_ptr.get_trainer_busy.remote():
            print("QUEUE_LENGTH {} at time {}".format(queue_length, time.time_ns()))
            await self.storage_ptr.set_trainer_busy.remote(True)
            return True
        await asyncio.sleep(2)
        # print("QUEUE_LENGTH_SLEEPY {} at time {}".format(queue_length, time.time_ns()))
        return False
