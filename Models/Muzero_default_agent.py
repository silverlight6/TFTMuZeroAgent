import config
from Models.MCTS_Util import *
from Models.abstract_model import AbstractNetwork
from Models.mlp_layers import mlp, MultiMlp


class MuZeroDefaultNetwork(AbstractNetwork):
    def __init__(self):
        super().__init__()
        self.full_support_size = config.ENCODER_NUM_STEPS

        self.representation_network = mlp(config.OBSERVATION_SIZE, [config.LAYER_HIDDEN_SIZE] *
                                          config.N_HEAD_HIDDEN_LAYERS, config.HIDDEN_STATE_SIZE)

        self.action_encodings = mlp(len(config.CHAMPION_ACTION_DIM), [
                                    config.LAYER_HIDDEN_SIZE] * 0, config.HIDDEN_STATE_SIZE)

        self.dynamics_hidden_state_network = torch.nn.LSTM(input_size=config.HIDDEN_STATE_SIZE,
                                                           num_layers=config.NUM_RNN_CELLS,
                                                           hidden_size=config.LSTM_SIZE,
                                                           batch_first=True).to(config.DEVICE)

        self.dynamics_reward_network = mlp(config.HIDDEN_STATE_SIZE, [
                                           1] * 1, self.full_support_size)

        self.prediction_policy_network = MultiMlp(config.HIDDEN_STATE_SIZE, [config.LAYER_HIDDEN_SIZE] *
                                                  config.N_HEAD_HIDDEN_LAYERS, config.CHAMPION_ACTION_DIM)

        self.prediction_value_network = mlp(config.HIDDEN_STATE_SIZE, [config.LAYER_HIDDEN_SIZE] *
                                            config.N_HEAD_HIDDEN_LAYERS, self.full_support_size)

        encoder_min = torch.tensor(-300., dtype=torch.float32)
        encoder_max = torch.tensor(300., dtype=torch.float32)
        self.value_encoder = ValueEncoder(*tuple(map(inverse_contractive_mapping, (encoder_min, encoder_max))), 0)
        self.reward_encoder = ValueEncoder(*tuple(map(inverse_contractive_mapping, (encoder_min, encoder_max))), 0)

    def prediction(self, encoded_state):
        policy_logits = self.prediction_policy_network(encoded_state)
        value = self.prediction_value_network(encoded_state)
        return policy_logits, value

    def representation(self, observation):
        observation = torch.from_numpy(observation).float().to(config.DEVICE)
        encoded_state = self.representation_network(observation)
        # Scale encoded state between [0, 1] (See appendix paper Training)
        min_encoded_state = encoded_state.min(1, keepdim=True)[0]
        max_encoded_state = encoded_state.max(1, keepdim=True)[0]
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
        encoded_state_normalized = (
            encoded_state - min_encoded_state
        ) / scale_encoded_state
        return encoded_state_normalized

    def dynamics(self, hidden_state, action):
        action = torch.from_numpy(action).to(config.DEVICE).to(torch.float32)

        action_encodings = self.action_encodings(action)

        lstm_state = self.flat_to_lstm_input(hidden_state)

        inputs = action_encodings
        inputs = inputs[:, None, :]

        h0, c0 = list(zip(*lstm_state))
        _, new_nested_states = self.dynamics_hidden_state_network(inputs,
                                                                  (torch.stack(h0, dim=0), torch.stack(c0, dim=0)))

        next_hidden_state = self.rnn_to_flat(
            new_nested_states)  # (8, 1024) ##DOUBLE CHECK THIS

        reward = self.dynamics_reward_network(next_hidden_state)

        # Scale encoded state between [0, 1] (See paper appendix Training)
        min_next_hidden_state = next_hidden_state.min(1, keepdim=True)[0]
        max_next_hidden_state = next_hidden_state.max(1, keepdim=True)[0]
        scale_next_hidden_state = max_next_hidden_state - min_next_hidden_state
        scale_next_hidden_state[scale_next_hidden_state < 1e-5] += 1e-5
        next_hidden_state_normalized = (
            next_hidden_state - min_next_hidden_state
        ) / scale_next_hidden_state

        return next_hidden_state_normalized, reward

    def initial_inference(self, observation):
        hidden_state = self.representation(observation)
        policy_logits, value_logits = self.prediction(hidden_state)

        value = self.value_encoder.decode_softmax(value_logits)

        reward = torch.zeros(observation.shape[0])
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
        """Maps LSTM state to flat vector."""
        states = []
        for cell_state in state:
            states.extend(cell_state)
        return torch.cat(states, dim=-1)

    @staticmethod
    def flat_to_lstm_input(state):
        """Maps flat vector to LSTM state."""
        tensors = []
        cur_idx = 0
        for size in config.RNN_SIZES:
            states = (state[Ellipsis, cur_idx:cur_idx + size],
                      state[Ellipsis, cur_idx + size:cur_idx + 2 * size])

            cur_idx += 2 * size
            tensors.append(states)
        # assert cur_idx == state.shape[-1]
        return tensors

    def recurrent_inference(self, hidden_state, action):
        next_hidden_state, reward_logits = self.dynamics(hidden_state, action)
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
