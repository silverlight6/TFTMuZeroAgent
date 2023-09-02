import torch
import config
import Models.MCTS_Util as utils
from Models.abstract_model import AbstractNetwork
from Models.torch_layers import mlp, MultiMlp, resnet


class MuZeroNetwork(AbstractNetwork):
    def __init__(self):
        super().__init__()
        self.full_support_size = config.ENCODER_NUM_STEPS

        self.representation_network = RepNetwork(
            [config.LAYER_HIDDEN_SIZE] * config.N_HEAD_HIDDEN_LAYERS,
            config.HIDDEN_STATE_SIZE
        )

        self.dynamics_network = DynNetwork(input_size=config.HIDDEN_STATE_SIZE,
                                                        num_layers=config.NUM_RNN_CELLS,
                                                        hidden_size=config.LSTM_SIZE,
                                                    )

        self.prediction_policy_network = MultiMlp(config.HIDDEN_STATE_SIZE, [config.LAYER_HIDDEN_SIZE] *
                                                  config.N_HEAD_HIDDEN_LAYERS, config.POLICY_HEAD_SIZES)

        self.prediction_value_network = mlp(config.HIDDEN_STATE_SIZE, [config.LAYER_HIDDEN_SIZE] *
                                            config.N_HEAD_HIDDEN_LAYERS, self.full_support_size)

        map_min = torch.tensor(-300., dtype=torch.float32)
        map_max = torch.tensor(300., dtype=torch.float32)
        self.value_encoder = utils.ValueEncoder(
            *tuple(map(utils.inverse_contractive_mapping, (map_min, map_max))), 0)
        self.reward_encoder = utils.ValueEncoder(
            *tuple(map(utils.inverse_contractive_mapping, (map_min, map_max))), 0)

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
        action = torch.from_numpy(action).to(config.DEVICE).to(torch.int64)
        one_hot_action = torch.nn.functional.one_hot(
            action[:, 0], config.ACTION_DIM[0])
        one_hot_target_a = torch.nn.functional.one_hot(
            action[:, 1], config.ACTION_DIM[1])
        one_hot_target_b = torch.nn.functional.one_hot(
            action[:, 2], config.ACTION_DIM[1])

        action_one_hot = torch.cat(
            [one_hot_action, one_hot_target_a, one_hot_target_b], dim=-1).float()

        action_encodings = self.action_encodings(action_one_hot)

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


class PredNetwork(torch.nn.Module):
    def __init__(self) -> torch.nn.Module:
        super().__init__()

        self.conv_value = torch.nn.Conv2d(256, 3, 1)
        self.bn_value = torch.nn.BatchNorm2d(3)
        self.conv_policy = torch.nn.Conv2d(256, 3, 1)
        self.bn_policy = torch.nn.BatchNorm2d(3)
        self.relu = torch.nn.ReLU(inplace=True)
        self.fc_internal_v = torch.nn.Linear(84, 128)
        self.fc_value = mlp(128, [config.LAYER_HIDDEN_SIZE] * config.N_HEAD_HIDDEN_LAYERS, config.ENCODER_NUM_STEPS)
        self.fc_internal_p = torch.nn.Linear(84, 128)
        self.fc_policy = MultiMlp(128, [config.LAYER_HIDDEN_SIZE] * config.N_HEAD_HIDDEN_LAYERS,
                                  config.POLICY_HEAD_SIZES, output_activation=torch.nn.Sigmoid)

    def forward(self, x):
        x = self.resnet(x)

        value = self.conv_value(x)
        value = self.bn_value(value)
        value = self.relu(value)
        value = torch.flatten(value, start_dim=1)
        # print("VALUE", value.shape)
        value = self.fc_internal_v(value)
        value = self.relu(value)
        value = self.fc_value(value)

        policy = self.conv_policy(x)
        policy = self.bn_policy(policy)
        policy = self.relu(policy)
        policy = torch.flatten(policy, start_dim=1)
        policy = self.fc_internal_p(policy)
        policy = self.relu(policy)
        # print("Policy", policy.shape)
        policy = self.fc_policy(policy)

        return policy, value


class RepNetwork(torch.nn.Module):
    def __init__(self, layer_sizes, output_size) -> torch.nn.Module:
        super().__init__()
        
        def feature_encoder(input_size):
            return mlp(input_size, layer_sizes, output_size)
        
        self.shop_encoder = feature_encoder(10)
        self.board_encoder = feature_encoder(10)
        self.bench_encoder = feature_encoder(10)
        self.states_encoder = feature_encoder(10)
        self.game_comp_encoder = feature_encoder(10)
        self.other_players_encoder = feature_encoder(10)

    def forward(self, x):
        shop = self.shop_encoder(x["shop"])
        board = self.board_encoder(x["board"])
        bench = self.bench_encoder(x["bench"])
        states = self.states_encoder(x["states"])
        game_comp = self.game_comp_encoder(x["game_comp"])
        other_players = self.other_players_encoder(x["other_players"])

        return {
            "shop": shop,
            "board": board,
            "bench": bench,
            "states": states,
            "game_comp": game_comp, 
            "other_players": other_players
        }


class DynNetwork(torch.nn.Module):
    def __init__(self, input_size, num_layers, hidden_size) -> torch.nn.Module:
        super().__init__()

        def lstm():
            return torch.nn.LSTM(input_size,
                                num_layers,
                                hidden_size,
                                batch_first=True).to(config.DEVICE)
        
        self.action_encodings = mlp(config.ACTION_CONCAT_SIZE, [
                            config.LAYER_HIDDEN_SIZE] * 0, config.HIDDEN_STATE_SIZE)

        self.dynamics_hidden_state_dict = torch.nn.ModuleDict({
            "shop": lstm(),
            "board": lstm(),
            "bench": lstm(),
            "states": lstm(),
            "game_comp": lstm(),
            "other_players": lstm()
        })

        self.dynamics_reward_network = mlp(hidden_size * 6, [1] * 1, self.full_support_size)

    def forward(self, x, action):
        action = torch.from_numpy(action).to(config.DEVICE).to(torch.int64)
        one_hot_action = torch.nn.functional.one_hot(
            action[:, 0], config.ACTION_DIM[0])
        one_hot_target_a = torch.nn.functional.one_hot(
            action[:, 1], config.ACTION_DIM[1])
        one_hot_target_b = torch.nn.functional.one_hot(
            action[:, 2], config.ACTION_DIM[1])

        action_one_hot = torch.cat(
            [one_hot_action, one_hot_target_a, one_hot_target_b], dim=-1).float()

        action_encodings = self.action_encodings(action_one_hot)

        inputs = action_encodings
        inputs = inputs[:, None, :]

        new_states = {}

        for key, hidden_state in x.items():
            lstm_state = self.flat_to_lstm_input(hidden_state)
            h0, c0 = list(zip(*lstm_state))
            _, new_nested_states = self.dynamics_hidden_state_dict[key](inputs,
                                                                        (torch.stack(h0, dim=0), torch.stack(c0, dim=0)))
            
            next_hidden_state = self.rnn_to_flat(
                    new_nested_states)  # (8, 1024) ##DOUBLE CHECK THIS
            
            min_next_hidden_state = next_hidden_state.min(1, keepdim=True)[0]
            max_next_hidden_state = next_hidden_state.max(1, keepdim=True)[0]
            scale_next_hidden_state = max_next_hidden_state - min_next_hidden_state
            scale_next_hidden_state[scale_next_hidden_state < 1e-5] += 1e-5
            next_hidden_state_normalized = (
                next_hidden_state - min_next_hidden_state
            ) / scale_next_hidden_state

            new_states[key] = next_hidden_state_normalized
            
        reward_input = torch.cat([
            new_states["shop"],
            new_states["board"],
            new_states["bench"],
            new_states["states"],
            new_states["game_comp"],
            new_states["other_players"]
        ], dim=-1)

        reward = self.dynamics_reward_network(reward_input)

        return new_states, reward

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

    @staticmethod
    def rnn_to_flat(state):
        """Maps LSTM state to flat vector."""
        states = []
        for cell_state in state:
            states.extend(cell_state)
        return torch.cat(states, dim=-1)