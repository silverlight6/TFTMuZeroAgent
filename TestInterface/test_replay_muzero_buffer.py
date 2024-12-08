import numpy as np
import config
from TestInterface.test_global_buffer import GlobalBuffer


class ReplayBuffer:
    def __init__(self, g_buffer: GlobalBuffer):
        self.gameplay_experiences = []
        self.rewards = []
        self.policy_distributions = []
        self.string_samples = []
        self.action_history = []
        self.root_values = []
        self.team_tiers = []
        self.team_champions = []
        self.g_buffer = g_buffer
        self.ending_position = -1

        if config.GUMBEL:
            self.buffer_config = {
                "data_keys": ["action", "policy_mask", "value", "reward", "policy", "priority"],
                "processors": {
                    "action": lambda idx, val, rc, reg: self.action_history[idx] if reg else 0,
                    "value": lambda idx, val, rc, reg: val,
                    "reward": lambda idx, val, rc, reg: rc[idx],
                    "policy": lambda idx, val, rc, reg: self.policy_distributions[idx],
                },
                "default_values": {
                    "action": 0,
                    "value": 0.0,
                    "reward": 0.0,
                    "policy": self.policy_distributions[0],
                },
                "return_keys": ["action", "policy_mask", "value", "reward", "policy"],
            }
        elif config.MUZERO_POSITION:
            self.buffer_config = {
                "data_keys": ["action", "policy_mask", "value", "policy"],
                "processors": {
                    "action": lambda idx, val, rc, reg: self.action_history[idx] if reg else 28,
                    "value": lambda idx, val, rc, reg: val,
                    "policy": lambda idx, val, rc, reg: self.policy_distributions[idx],
                },
                "default_values": {
                    "action": 28,
                    "value": 0.0,
                    "policy": self.policy_distributions[0],
                },
                "return_keys": ["action", "policy_mask", "value", "policy"],
            }
        else:
            self.buffer_config = {
                "data_keys": [
                    "action", "value_mask", "reward_mask", "policy_mask",
                    "value", "reward", "policy", "sample", "tier",
                    "final_tier", "champion", "priority"
                ],
                "processors": {
                    "action": lambda idx, val, rc, reg: self.action_history[idx] if reg else [0, 0, 0],
                    "value": lambda idx, val, rc, reg: val,
                    "reward": lambda idx, val, rc, reg: rc[idx],
                    "policy": lambda idx, val, rc, reg: self.policy_distributions[idx],
                },
                "default_values": {
                    "action": [0, 0, 0],
                    "value": 0.0,
                    "reward": 0.0,
                    "policy": self.policy_distributions[0],
                },
                "return_keys": [
                    "action", "value_mask", "reward_mask", "policy_mask",
                    "value", "reward", "policy", "sample", "tier",
                    "final_tier", "champion"
                ],
            }

    def store_buffer(self, observation, action, reward, policy, root_value, **optional_data):
        """
        Generalized method to store gameplay experience in the buffer.

        Args:
            observation: Observed state of the environment.
            action: Action taken in the environment.
            reward: Reward received from the environment.
            policy: Policy distribution for the current state.
            root_value: Root value for the current state.
            **optional_data: Additional data fields to be stored in respective buffers.
        """
        # Always store core data
        self.gameplay_experiences.append(observation)
        self.action_history.append(action)
        np.clip(reward, config.MINIMUM_REWARD, config.MAXIMUM_REWARD)
        self.rewards.append(reward)
        self.policy_distributions.append(policy)
        self.root_values.append(root_value)

        # Dynamically store optional data
        for key, value in optional_data.items():
            if hasattr(self, key):  # Ensure the attribute exists in the class
                getattr(self, key).append(value)
            else:
                raise AttributeError(f"Buffer '{key}' does not exist in the class.")

    # I realize this can be a 2 liner with for attribute in list -> setattr, but it makes my pycharm
    # go a little crazy so leaving it like this.
    def reset(self):
        self.gameplay_experiences = []
        self.rewards = []
        self.policy_distributions = []
        self.string_samples = []
        self.action_history = []
        self.root_values = []
        self.team_tiers = []
        self.team_champions = []

    def get_prev_action(self):
        if self.action_history:
            return self.action_history[-1]
        else:
            return 9

    def get_reward_sequence(self):
        return self.rewards

    def set_reward_sequence(self, rewards):
        self.rewards = rewards

    def get_ending_position(self):
        return self.ending_position

    def set_ending_position(self, ending_position):
        self.ending_position = ending_position

    def store_global_buffer(self):
        """
        Generalized method for storing global buffers with configurable data processing.
        """
        # Determine the number of samples per player
        samples_per_player = (
            config.SAMPLES_PER_PLAYER
            if (len(self.gameplay_experiences) - config.UNROLL_STEPS) > config.SAMPLES_PER_PLAYER
            else len(self.gameplay_experiences) - config.UNROLL_STEPS
        )
        if samples_per_player < config.UNROLL_STEPS and len(self.gameplay_experiences) > 0:
            samples_per_player = len(self.gameplay_experiences)

        if samples_per_player > 0:
            samples = range(0, len(self.gameplay_experiences))
            num_steps = len(self.gameplay_experiences)
            reward_correction = []
            prev_reward = 0
            output_sample_set = []

            # Compute reward corrections
            for reward in self.rewards:
                reward_correction.append(reward - prev_reward)
                prev_reward = reward

            for sample in samples:
                # Initialize sets for the sample
                data_sets = {key: [] for key in self.buffer_config["data_keys"]}

                for current_index in range(sample, sample + config.UNROLL_STEPS + 1):
                    # Calculate bootstrapped value
                    bootstrap_index = (
                        current_index + config.TD_STEPS
                        if config.TD_STEPS > 0
                        else len(reward_correction)
                    )
                    value = (
                        self.root_values[bootstrap_index] * config.DISCOUNT ** config.TD_STEPS
                        if config.TD_STEPS > 0 and bootstrap_index < len(self.root_values)
                        else 0.0
                    )
                    for i, reward_corrected in enumerate(reward_correction[current_index:bootstrap_index]):
                        value += reward_corrected * config.DISCOUNT ** i

                    # Handle absorbing states and terminal conditions
                    if current_index < num_steps - 1:
                        self._process_step(data_sets, current_index, value, reward_correction, True)
                    elif current_index == num_steps - 1:
                        self._process_step(data_sets, current_index, value, reward_correction, False)
                    else:
                        self._process_absorbing_state(data_sets)

                # Add priority and store sample
                priority = self._calculate_priority(data_sets["priority"])
                output_sample_set.append(
                    [[self.gameplay_experiences[sample]] + [data_sets[key] for key in self.buffer_config["return_keys"]],
                     priority]
                )

            self.g_buffer.store_replay_sequence([output_sample_set, self.ending_position])

    def _process_step(self, data_sets, current_index, value, reward_correction, is_regular):
        """Helper method to process a regular or terminal step."""
        for key, processor in self.buffer_config["processors"].items():
            data_sets[key].append(processor(current_index, value, reward_correction, is_regular))

    def _process_absorbing_state(self, data_sets):
        """Helper method to process an absorbing state."""
        for key in self.buffer_config["default_values"]:
            data_sets[key].append(self.buffer_config["default_values"][key])

    @staticmethod
    def _calculate_priority(priority_set):
        """Helper method to calculate priority for a sample."""
        priority = priority_set[0]
        div = -priority
        for i in priority_set:
            div += i
        return 1 / (priority / div) + np.random.rand() * 0.00001

    def print_reward(self):
        print(f"rewards for buffer {self.rewards}")
