import numpy as np
from Models.MCTS_torch import MCTS

class Default_MCTS(MCTS):
    def __init__(self, network, config):
        super().__init__(network, config)
        self.default_string_mapping = []
        self.champ_decider_action_dim = config.CHAMP_DECIDER_ACTION_DIM

    @staticmethod
    def encode_action_to_str(policy_logits, mask):
        return policy_logits, []

    # TODO: Duplication value and shink size of array with duplicates.
    # I don't expect duplicates too often with an action space size of 2^57 but it's possible.
    def sample(self, policy_logits, string_mapping, num_samples):
        batch_size = len(policy_logits[0])  # 8

        output_logits = []
        output_string_mapping = []
        output_byte_mapping = []
        policy_sizes = []

        for idx in range(batch_size):
            local_byte = []
            sampled_action = []
            probs = self.softmax_stable(policy_logits[0][idx])
            policy_range = np.arange(stop=len(policy_logits[0][idx]))

            samples = np.random.choice(a=policy_range, p=probs, size=num_samples)
            for sample in samples:
                sampled_action.append(str(sample))

            for i in range(1, len(self.champ_decider_action_dim)):
                probs = self.softmax_stable(policy_logits[i][idx])
                policy_range = np.arange(stop=len(policy_logits[i][idx]))

                samples = np.random.choice(a=policy_range, p=probs, size=num_samples)

                for k, sample in enumerate(samples):
                    sampled_action[k] = sampled_action[k] + "_" + str(sample)

            for sample in enumerate(sampled_action):
                local_byte.append(bytes(str(sample[1]), "utf-8"))

            output_logits.append([((1 / num_samples) * num_samples)])
            output_string_mapping.append(sampled_action)
            output_byte_mapping.append(local_byte)
            policy_sizes.append(num_samples)

        return output_logits, output_string_mapping, output_byte_mapping, policy_sizes
