import torch
import config
import Models.MCTS_Util as utils
from Models.abstract_model import AbstractNetwork
from Models.torch_layers import mlp, MultiMlp, resnet, Normalize, MemoryLayer


class MuZeroNetwork(AbstractNetwork):
    def __init__(self):
        super().__init__()
        self.full_support_size = config.ENCODER_NUM_STEPS

        self.representation_network = RepNetwork(
            [config.LAYER_HIDDEN_SIZE] * config.N_HEAD_HIDDEN_LAYERS,
            config.HIDDEN_STATE_SIZE
        )

        self.dynamics_network = DynNetwork(input_size=config.HIDDEN_STATE_SIZE, num_layers=config.NUM_RNN_CELLS,
                                           hidden_size=config.LSTM_SIZE)

        self.prediction_network = PredNetwork()

        map_min = torch.tensor(-300., dtype=torch.float32)
        map_max = torch.tensor(300., dtype=torch.float32)
        self.value_encoder = utils.ValueEncoder(
            *tuple(map(utils.inverse_contractive_mapping, (map_min, map_max))), 0)
        self.reward_encoder = utils.ValueEncoder(
            *tuple(map(utils.inverse_contractive_mapping, (map_min, map_max))), 0)

    def prediction(self, encoded_state):
        policy_logits, value = self.prediction_network(encoded_state)
        return policy_logits, value

    def representation(self, observation):
        observation = {label: torch.from_numpy(value).float().to(config.DEVICE) for label, value in observation.items()}
        return self.representation_network(observation)

    def dynamics(self, x, action):
        return self.dynamics_network(x, action)

    def initial_inference(self, observation):
        hidden_state = self.representation(observation)
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

        def feature_encoder(input_size, layer_sizes, output_size):
            return mlp(input_size, layer_sizes, output_size)

        # 6 is the number of separate dynamic networks
        self.prediction_value_network = feature_encoder(config.HIDDEN_STATE_SIZE * 6,
                                                        [config.LAYER_HIDDEN_SIZE] * config.N_HEAD_HIDDEN_LAYERS,
                                                        config.ENCODER_NUM_STEPS)
        self.decision_network = feature_encoder(config.HIDDEN_STATE_SIZE * 6,
                                                [config.LAYER_HIDDEN_SIZE] * config.N_HEAD_HIDDEN_LAYERS,
                                                config.POLICY_HEAD_SIZES[0])
        # Just shop
        self.shop_action_network = feature_encoder(config.HIDDEN_STATE_SIZE + config.POLICY_HEAD_SIZES[0],
                                                   [config.LAYER_HIDDEN_SIZE] * config.N_HEAD_HIDDEN_LAYERS,
                                                   config.POLICY_HEAD_SIZES[1])
        # Board, Bench, Comp, Other player
        self.movement_action_network = feature_encoder(config.HIDDEN_STATE_SIZE * 4 + config.POLICY_HEAD_SIZES[0],
                                                       [config.LAYER_HIDDEN_SIZE] * config.N_HEAD_HIDDEN_LAYERS,
                                                       config.POLICY_HEAD_SIZES[2])
        # Game_state, Board, Comp
        self.item_action_network = feature_encoder(config.HIDDEN_STATE_SIZE * 3 + config.POLICY_HEAD_SIZES[0],
                                                   [config.LAYER_HIDDEN_SIZE] * config.N_HEAD_HIDDEN_LAYERS,
                                                   config.POLICY_HEAD_SIZES[3])
        # Bench, Game_State
        self.sell_action_network = feature_encoder(config.HIDDEN_STATE_SIZE * 2 + config.POLICY_HEAD_SIZES[0],
                                                   [config.LAYER_HIDDEN_SIZE] * config.N_HEAD_HIDDEN_LAYERS,
                                                   config.POLICY_HEAD_SIZES[4])

    def forward(self, x):
        value_decision_input = torch.cat([
            x["shop"],
            x["board"],
            x["bench"],
            x["states"],
            x["game_comp"],
            x["other_players"]
        ], dim=-1)

        value = self.prediction_value_network(value_decision_input)
        decision = self.decision_network(value_decision_input)

        shop_input = torch.cat([
            decision,
            x["shop"]
        ], 1)
        shop = self.shop_action_network(shop_input)

        movement_input = torch.cat([
            decision,
            x["board"],
            x["bench"],
            x["game_comp"],
            x["other_players"]
        ], 1)
        movement = self.movement_action_network(movement_input)

        item_input = torch.cat([
            decision,
            x["states"],
            x["board"],
            x["game_comp"]
        ], 1)
        item = self.item_action_network(item_input)

        sell_input = torch.cat([
            decision,
            x["bench"],
            x["states"]
        ], 1)
        sell = self.sell_action_network(sell_input)

        return [decision, shop, movement, item, sell], value


class RepNetwork(torch.nn.Module):
    def __init__(self, layer_sizes, output_size) -> torch.nn.Module:
        super().__init__()
        
        def feature_encoder(input_size):
            return torch.nn.Sequential(
                mlp(input_size, layer_sizes, output_size),
                Normalize()
            ).to(config.DEVICE)
        
        self.shop_encoder = feature_encoder(config.SHOP_INPUT_SIZE)
        self.board_encoder = feature_encoder(config.BOARD_INPUT_SIZE)
        self.bench_encoder = feature_encoder(config.BENCH_INPUT_SIZE)
        self.states_encoder = feature_encoder(config.STATE_INPUT_SIZE)
        self.game_comp_encoder = feature_encoder(config.COMP_INPUT_SIZE)
        self.other_players_encoder = feature_encoder(config.OTHER_PLAYER_INPUT_SIZE)

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

        def memory():
            return torch.nn.Sequential(
                MemoryLayer(input_size, num_layers, hidden_size),
                Normalize()
            ).to(config.DEVICE)
        
        self.action_encodings = mlp(config.ACTION_CONCAT_SIZE, [
                            config.LAYER_HIDDEN_SIZE] * 0, config.HIDDEN_STATE_SIZE)

        self.dynamics_memory = torch.nn.ModuleDict({
            "shop": memory(),
            "board": memory(),
            "bench": memory(),
            "states": memory(),
            "game_comp": memory(),
            "other_players": memory()
        })

        self.dynamics_reward_network = mlp(input_size * 6, [1] * 1, config.ENCODER_NUM_STEPS)

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

        action_encoding = self.action_encodings(action_one_hot)

        inputs = action_encoding
        inputs = inputs[:, None, :]

        new_states = {}

        for key, hidden_state in x.items():
            new_states[key] = self.dynamics_memory[key]((inputs, hidden_state))
            
        reward_input = torch.cat([
            new_states["shop"],
            new_states["board"],
            new_states["bench"],
            new_states["states"],
            new_states["game_comp"],
            new_states["other_players"]
        ], dim=1)

        reward = self.dynamics_reward_network(reward_input)

        return new_states, reward