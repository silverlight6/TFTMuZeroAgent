import torch
import Core.MCTS_Trees.MCTS_Util as utils
import config
from Core.TorchModels.abstract_model import AbstractNetwork
from Core.TorchComponents.representation_models import RepEmbeddingNetwork
from Core.TorchComponents.dynamics_models import DynNetwork
from Core.TorchComponents.prediction_models import PredNetwork


class MuZeroNetwork(AbstractNetwork):
    def __init__(self, model_config):
        super().__init__()
        self.full_support_size = model_config.ENCODER_NUM_STEPS

        self.representation_network = RepEmbeddingNetwork(model_config)

        self.dynamics_network = DynNetwork(input_size=model_config.HIDDEN_STATE_SIZE,
                                           num_layers=model_config.NUM_RNN_CELLS,
                                           hidden_size=model_config.LSTM_SIZE, model_config=model_config)

        self.prediction_network = PredNetwork(model_config)

        map_min = torch.tensor(-300., dtype=torch.float32)
        map_max = torch.tensor(300., dtype=torch.float32)
        self.value_encoder = utils.ValueEncoder(
            *tuple(map(utils.inverse_contractive_mapping, (map_min, map_max))), 0)
        self.reward_encoder = utils.ValueEncoder(
            *tuple(map(utils.inverse_contractive_mapping, (map_min, map_max))), 0)

        self.model_config = model_config

    def prediction(self, encoded_state, training=False):
        if training:
            policy_logits, value, comp, final_comp, champ = self.prediction_network(encoded_state, training)
            return policy_logits, value, comp, final_comp, champ
        else:
            policy_logits, value = self.prediction_network(encoded_state, training)
            return policy_logits, value

    def representation(self, observation):
        observation = {label: torch.from_numpy(value).to(config.DEVICE) for label, value in observation.items()}
        return self.representation_network(observation)

    def dynamics(self, x, action):
        return self.dynamics_network(x, action)

    def initial_inference(self, observation, training=False):
        hidden_state = self.representation(observation)
        if training:
            policy_logits, value_logits, comp, final_comp, champ = self.prediction(hidden_state, training)

            value = self.value_encoder.decode_softmax(value_logits)

            reward = torch.zeros(observation["shop"].shape[0])
            reward_logits = self.reward_encoder.encode(reward)

            outputs = {
                "value": value,
                "value_logits": value_logits,
                "reward": reward,
                "reward_logits": reward_logits,
                "policy_logits": policy_logits,
                "hidden_state": hidden_state,
                "comp": comp,
                "final_comp": final_comp,
                "champ": champ
            }

        else:
            policy_logits, value_logits = self.prediction(hidden_state)

            value = self.value_encoder.decode_softmax(value_logits)

            reward = torch.zeros(observation["shop"].shape[0])
            reward_logits = self.reward_encoder.encode(reward)

            outputs = {
                "value": value,
                "value_logits": value_logits,
                "reward": reward,
                "reward_logits": reward_logits,
                "policy_logits": policy_logits,
                "hidden_state": hidden_state
            }
        return outputs

    @staticmethod
    def rnn_to_flat(state):
        """Maps LSTM state to flat token."""
        states = []
        for cell_state in state:
            states.extend(cell_state)
        return torch.cat(states, dim=-1)

    def flat_to_lstm_input(self, state):
        """Maps flat token to LSTM state."""
        tensors = []
        cur_idx = 0
        for size in self.model_config.RNN_SIZES:
            states = (state[Ellipsis, cur_idx:cur_idx + size],
                      state[Ellipsis, cur_idx + size:cur_idx + 2 * size])

            cur_idx += 2 * size
            tensors.append(states)
        # assert cur_idx == state.shape[-1]
        return tensors

    def recurrent_inference(self, hidden_state, action, training=False):
        next_hidden_state, reward_logits = self.dynamics(hidden_state, action)
        if training:
            policy_logits, value_logits, comp, final_comp, champ = self.prediction(next_hidden_state, training)
            value = self.value_encoder.decode_softmax(value_logits)
            reward = self.reward_encoder.decode_softmax(reward_logits)
            outputs = {
                "value": value,
                "value_logits": value_logits,
                "reward": reward,
                "reward_logits": reward_logits,
                "policy_logits": policy_logits,
                "hidden_state": next_hidden_state,
                "comp": comp,
                "final_comp": final_comp,
                "champ": champ
            }
        else:
            policy_logits, value_logits = self.prediction(next_hidden_state)
            value = self.value_encoder.decode_softmax(value_logits)
            reward = self.reward_encoder.decode_softmax(reward_logits)
            outputs = {
                "value": value,
                "value_logits": value_logits,
                "reward": reward,
                "reward_logits": reward_logits,
                "policy_logits": policy_logits,
                "hidden_state": next_hidden_state
            }
        return outputs
