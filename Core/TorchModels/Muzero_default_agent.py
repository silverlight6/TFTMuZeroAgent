import torch
import config
import Core.MCTS_Trees.MCTS_Util as utils
from Core.TorchModels.abstract_model import AbstractNetwork
from Core.TorchComponents.torch_layers import mlp, MultiMlp, Normalize, MemoryLayer


class MuZeroDefaultNetwork(AbstractNetwork):
    def __init__(self, model_config):
        super().__init__()
        self.full_support_size = model_config.ENCODER_NUM_STEPS

        self.representation_network = RepNetwork(
            [model_config.LAYER_HIDDEN_SIZE // 2] * model_config.N_HEAD_HIDDEN_LAYERS,
            model_config.HIDDEN_STATE_SIZE // 2, model_config
        )

        self.dynamics_network = DynNetwork(input_size=model_config.HIDDEN_STATE_SIZE,
                                           num_layers=model_config.NUM_RNN_CELLS,
                                           hidden_size=model_config.LSTM_SIZE,
                                           model_config=model_config)

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
            policy_logits, value, final_comp = self.prediction_network(encoded_state, training)
            return policy_logits, value, final_comp

    def representation(self, observation):
        observation = {label: torch.from_numpy(value).float().to(config.DEVICE) for label, value in observation.items()}
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
            policy_logits, value_logits, final_comp = self.prediction(hidden_state)

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
                "final_comp": final_comp
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
            policy_logits, value_logits, _ = self.prediction(next_hidden_state)
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
    def __init__(self, model_config) -> torch.nn.Module:
        super().__init__()

        def feature_encoder(input_size, layer_sizes, output_size):
            return torch.nn.Sequential(mlp(input_size, layer_sizes, output_size)).to(config.DEVICE)

        # 6 is the number of separate dynamic networks
        self.prediction_value_network = feature_encoder(model_config.HIDDEN_STATE_SIZE,
                                                        [model_config.LAYER_HIDDEN_SIZE] * model_config.N_HEAD_HIDDEN_LAYERS,
                                                        model_config.ENCODER_NUM_STEPS)

        # This includes champion_list, sell_chosen, item_choice
        self.default_guide_network = MultiMlp(model_config.HIDDEN_STATE_SIZE,
                                              [model_config.LAYER_HIDDEN_SIZE] * model_config.N_HEAD_HIDDEN_LAYERS,
                                              config.CHAMP_DECIDER_ACTION_DIM)

        self.comp_predictor_network = MultiMlp(model_config.HIDDEN_STATE_SIZE,
                                               [model_config.LAYER_HIDDEN_SIZE] * model_config.N_HEAD_HIDDEN_LAYERS,
                                               config.TEAM_TIERS_VECTOR, output_activation=torch.nn.ReLU)

        self.final_comp_predictor_network = MultiMlp(model_config.HIDDEN_STATE_SIZE,
                                                     [model_config.LAYER_HIDDEN_SIZE] * model_config.N_HEAD_HIDDEN_LAYERS,
                                                     config.TEAM_TIERS_VECTOR, output_activation=torch.nn.ReLU)

        self.champ_predictor_network = MultiMlp(model_config.HIDDEN_STATE_SIZE,
                                                [model_config.LAYER_HIDDEN_SIZE] * model_config.N_HEAD_HIDDEN_LAYERS,
                                                config.CHAMPION_LIST_DIM, output_activation=torch.nn.ReLU)

    def forward(self, hidden_state, training=True):

        value = self.prediction_value_network(hidden_state)

        default_guide = self.default_guide_network(hidden_state)

        final_comp = self.final_comp_predictor_network(hidden_state)

        if not training:
            return default_guide, value, final_comp
        else:
            comp = self.comp_predictor_network(hidden_state)

            champ = self.champ_predictor_network(hidden_state)
            return default_guide, value, comp, final_comp, champ


class RepNetwork(torch.nn.Module):
    def __init__(self, layer_sizes, output_size, model_config) -> torch.nn.Module:
        super().__init__()

        def feature_encoder(input_size, feature_layer_sizes, feature_output_sizes):
            return torch.nn.Sequential(
                mlp(input_size, feature_layer_sizes, feature_output_sizes),
                Normalize()
            ).to(config.DEVICE)

        self.scalar_encoder = feature_encoder(config.SCALAR_INPUT_SIZE, layer_sizes, output_size)
        self.shop_encoder = feature_encoder(config.SHOP_INPUT_SIZE, layer_sizes, output_size)
        self.board_encoder = feature_encoder(config.BOARD_INPUT_SIZE, layer_sizes, output_size)
        self.bench_encoder = feature_encoder(config.BENCH_INPUT_SIZE, layer_sizes, output_size)
        self.items_encoder = feature_encoder(config.ITEMS_INPUT_SIZE, layer_sizes, output_size)
        self.traits_encoder = feature_encoder(config.TRAIT_INPUT_SIZE, layer_sizes, output_size)
        self.other_players_encoder = feature_encoder(config.OTHER_PLAYER_INPUT_SIZE, layer_sizes, output_size)

        self.feature_to_hidden = feature_encoder((model_config.HIDDEN_STATE_SIZE // 2) * 7,
                                                 [model_config.LAYER_HIDDEN_SIZE] *
                                                 model_config.N_HEAD_HIDDEN_LAYERS,
                                                 model_config.HIDDEN_STATE_SIZE)

    def forward(self, x):
        scalar = self.scalar_encoder(x["scalars"])
        shop = self.shop_encoder(x["shop"])
        board = self.board_encoder(x["board"])
        bench = self.bench_encoder(x["bench"])
        items = self.items_encoder(x["items"])
        traits = self.traits_encoder(x["traits"])
        other_players = self.other_players_encoder(x["other_players"])

        full_state = torch.cat((scalar, shop, board, bench, items, traits, other_players), -1)

        hidden_state = self.feature_to_hidden(full_state)

        return hidden_state


class DynNetwork(torch.nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, model_config) -> torch.nn.Module:
        super().__init__()

        def memory():
            return torch.nn.Sequential(
                MemoryLayer(input_size, num_layers, hidden_size, model_config),
                Normalize()
            ).to(config.DEVICE)

        self.action_encodings = mlp(config.ACTION_CONCAT_SIZE, [
            model_config.LAYER_HIDDEN_SIZE] * 0, model_config.HIDDEN_STATE_SIZE)

        self.dynamics_memory = memory()

        self.dynamics_reward_network = mlp(input_size, [model_config.LAYER_HIDDEN_SIZE] * 1,
                                           model_config.ENCODER_NUM_STEPS)
        self.model_config = model_config

    def forward(self, hidden_state, action):
        action = torch.from_numpy(action).to(config.DEVICE).to(torch.int64)
        one_hot_action = torch.nn.functional.one_hot(action[:, 0], config.ACTION_DIM[0])
        one_hot_target_a = torch.nn.functional.one_hot(action[:, 1], config.ACTION_DIM[1])
        one_hot_target_b = torch.nn.functional.one_hot(action[:, 2], config.ACTION_DIM[1])

        action_one_hot = torch.cat([one_hot_action, one_hot_target_a, one_hot_target_b], dim=-1).float()

        action_encoding = self.action_encodings(action_one_hot)

        inputs = action_encoding
        inputs = inputs[:, None, :]

        new_hidden_state = self.dynamics_memory((inputs, hidden_state))

        reward = self.dynamics_reward_network(new_hidden_state)

        return new_hidden_state, reward