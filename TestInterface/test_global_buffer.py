import config
import numpy as np
import torch
from collections import deque

class GlobalBuffer:
    def __init__(self):
        self.gameplay_experiences = deque(maxlen=config.GLOBAL_BUFFER_SIZE)
        self.average_position = deque(maxlen=config.GLOBAL_BUFFER_SIZE)
        self.batch_size = config.BATCH_SIZE
        if config.MUZERO_POSITION:
            self.data_key = [
                "observation", "action_history", "value_mask", "policy_mask", "value", "policy"
            ]
        elif config.GUMBEL:
            self.data_keys = [
                "observation", "action_history", "policy_mask", "value", "reward", "policy"
            ]
        else:
            self.data_keys = [
                "observation", "action_history", "value_mask", "reward_mask", "policy_mask",
                "value", "reward", "policy", "sample_set", "tier_set", "final_tier_set", "champion_set"
            ]

    def sample_batch_general(self, process_position=False):
        """
        Generalized batch sampling method.

        Args:
            process_position (bool): Whether to include position data in processing.

        Returns:
            list: A prepared batch based on the specified data keys.
        """
        batch_data = {key: [] for key in self.data_keys}  # Dictionary to hold data for each key
        importance_weights = []

        for _ in range(self.batch_size):
            # Pop data from gameplay_experiences
            data, priority = self.gameplay_experiences.pop()

            # Assign data fields to respective keys
            for key, value in zip(self.data_keys, data):
                batch_data[key].append(value)

            # Handle position data if required
            if process_position:
                position, _ = self.average_position.pop()
                batch_data["position"].append(position)

            # Importance weights calculation
            importance_weights.append(1 / self.batch_size / priority)

        # Convert lists to appropriate formats
        for key in batch_data:
            if key == "observation":  # Example for specific processing
                batch_data[key] = torch.stack(batch_data[key]).to(config.DEVICE)
            else:
                batch_data[key] = np.asarray(batch_data[key]).astype('float32')

        # Process importance weights
        importance_weights = np.asarray(importance_weights).astype('float32')
        importance_weights /= np.max(importance_weights)

        # Add importance weights if required
        batch_data["importance_weights"] = importance_weights

        # Return data as list
        return [batch_data[key] for key in self.data_keys] + [importance_weights]

    def store_replay_sequence(self, samples):
        """
        Description:
            Stores data into the gameplay_experience buffer. Currently, not using prioritized replay buffer since
            we are not processing the data multiple times and all data gets sent to the trainer. Priority used
            to create importance weights.

        Args:
            samples (list): All samples from one game from one agent.
        """
        for sample in samples:
            self.gameplay_experiences.append(sample)

    def available_batch(self):
        queue_length = len(self.gameplay_experiences)
        if queue_length >= self.batch_size:
            return True
        return False

    def buffer_len(self):
        return len(self.gameplay_experiences)
