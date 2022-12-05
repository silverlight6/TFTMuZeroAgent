from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Dict, List
from datetime import datetime
from numba import jit
import collections
import math
import config
import time
import numpy as np
import tensorflow as tf

##########################
####### Helpers ##########

MAXIMUM_FLOAT_VALUE = float('inf')

KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])

NetworkOutput = collections.namedtuple(
    'NetworkOutput',
    'value reward policy_logits hidden_state')


class MinMaxStats(object):
    """A class that holds the min-max values of the tree."""

    def __init__(self, minimum: int, maximum: int):
        self.maximum = minimum
        self.minimum = maximum

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class Node(object):

    def __init__(self, prior: float):
        self.visit_count = 0
        self.to_play = 1
        self.prior = prior
        self.value_sum = 0
        # 5 because there are 5 separate actions.
        self.children = [{} for _ in range(5)]
        self.hidden_state = None
        self.reward = 0

    def expanded(self) -> bool:
        return len(self.children[0]) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


##### JITTED FUNCTIONS #######
@jit(target_backend='cuda', nopython=True)
def expand_node2(network_output, action_dim):
    policy = []
    for i, action_dim in enumerate(action_dim):
        policy.append({b: math.exp(network_output[i][0][b]) for b in range(action_dim)})
    return policy


class MuZero_agent(tf.Module):

    def __init__(self, t_board=None):
        super().__init__(name='MuZeroAgent')
        self.start_time = time.time_ns()
        self.ckpt_time = time.time_ns()
        # action_dim = [possible actions, item bench, unit bench, x axis, y axis]
        self.action_dim = [12, 10, 9, 7, 4]
        self.mlp_block_num = 2
        self.batch_size = config.BATCH_SIZE
        self.t_board = t_board
        logs = "./logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.file_writer = tf.summary.create_file_writer(logs)
        self.file_writer.set_as_default()
        self.head_hidden_sizes = [config.HEAD_HIDDEN_SIZE] * config.N_HEAD_HIDDEN_LAYERS
        self.num_actions = 0
        self.mlp1 = Mlp(hidden_size=config.HIDDEN_STATE_SIZE, mlp_dim=config.HEAD_HIDDEN_SIZE)
        self.mlp2 = Mlp(hidden_size=config.HIDDEN_STATE_SIZE, mlp_dim=config.HEAD_HIDDEN_SIZE)

        rnn_cell_cls = {
            'lstm': tf.keras.layers.LSTMCell,
        }['lstm']
        rnn_cells = [
            rnn_cell_cls(
                size,
                recurrent_activation='sigmoid',
                name='cell_{}'.format(idx)) for idx, size in enumerate([config.HIDDEN_STATE_SIZE])]
        self._core = tf.keras.layers.StackedRNNCells(rnn_cells, name='recurrent_core')

        # I can play around with the num_steps (0) later.
        # Not a 100% sure what difference it will make at this time.
        self.value_encoder = ValueEncoder(*tuple(map(inverse_contractive_mapping, (-300., 300.))), 0)

        self.reward_encoder = ValueEncoder(*tuple(map(inverse_contractive_mapping, (-300., 300.))), 0)

        self._to_hidden = tf.keras.layers.Dense(config.HIDDEN_STATE_SIZE, activation='sigmoid', name='final')
        self._value_head = tf.keras.layers.Dense(self.value_encoder.num_steps, name='output', dtype=tf.float32)
        self._reward_head = tf.keras.layers.Dense(self.reward_encoder.num_steps, name='output', dtype=tf.float32)

        self._shop_output = tf.keras.layers.Dense(self.action_dim[0], activation='softmax', name='shop_layer')
        self._item_output = tf.keras.layers.Dense(self.action_dim[1], activation='softmax', name='item_layer')
        self._bench_output = tf.keras.layers.Dense(self.action_dim[2], activation='softmax', name='bench_layer')
        self._board_output_x = tf.keras.layers.Dense(self.action_dim[3], activation='softmax', name='board_x_layer')
        self._board_output_y = tf.keras.layers.Dense(self.action_dim[4], activation='softmax', name='board_y_layer')

    def policy(self, observation, player_num):
        self.ckpt_time = time.time_ns()
        root = Node(0)
        # observation = np.expand_dims(observation, axis=0)
        network_output = self.initial_inference(observation)

        self.expand_node(root, player_num, network_output)
        self.add_exploration_noise(root)

        # We then run a Monte Carlo Tree Search using only action sequences and the
        # model learned by the network.
        initial_action = [
            np.argmax(network_output["policy_logits"][0].numpy()),
            np.argmax(network_output["policy_logits"][1].numpy()),
            np.argmax(network_output["policy_logits"][2].numpy()),
            np.argmax(network_output["policy_logits"][3].numpy()),
            np.argmax(network_output["policy_logits"][4].numpy())
        ]
        self.run_mcts(root, initial_action, player_num)
        action = self.select_action(root)

        # Masking only if training is based on the actions taken in the environment.
        # Training in MuZero is mostly based on the predicted actions rather than the real ones.
        # network_output["policy_logits"], action = self.maskInput(network_output["policy_logits"], action)

        # Notes on possibilities for other dimensions at the bottom
        self.num_actions += 1

        return action, network_output["policy_logits"]

    def expand_node(self, node: Node, to_play: int, network_output):
        self.ckpt = time.time_ns()
        node.to_play = to_play
        node.hidden_state = network_output["hidden_state"]
        node.reward = network_output["reward"]
        input_to_jitfunc = [] 
        for i in network_output["policy_logits"]:
            input_to_jitfunc.append(i.numpy())
        # print(input_to_jitfunc)
        policy = expand_node2(input_to_jitfunc, self.action_dim)
        for i, action_dim in enumerate(policy):
            policy_sum = sum(action_dim.values())
            for action, p in action_dim.items():
                node.children[i][action] = Node(p / policy_sum)

    # So let me make a few quick notes here first
    # I want to build blocks in other functions and then tie them together at the end.
    # I also want to start to put more stuff in the configuration as I go
    # The first place I should start to put things in is to encode observation.
    def initial_inference(self, observation) -> Dict:
        # representation + prediction function
        encoded_observation = self.encode_observation(
            observation)
        hidden_state = self.to_hidden(encoded_observation)

        # could add encoding but more research has to be done to why that is a good idea
        value_logits = self.value_head(hidden_state)
        value_logits = tf.nn.softmax(value_logits)
        value = self.value_encoder.decode(value_logits)

        # Rewards are only calculated in recurrent_inference.
        reward = tf.zeros_like(value)
        reward_logits = self.reward_encoder.encode(reward)

        policy_logits = self.policy_head(hidden_state)

        outputs = {
            "value": value,
            "value_logits": value_logits,
            "reward": reward,
            "reward_logits": reward_logits,
            "policy_logits": policy_logits,
            "hidden_state": hidden_state
        }
        return outputs

    def recurrent_inference(self, hidden_state, action) -> Dict:
        # dynamics + prediction function
        # only looking at the first dimension for now.
        # I am setting off values to 0 out of preference, Google sets it to 1. Not sure if there is a real difference.
        one_hot_action = tf.Variable(tf.one_hot(action[:, 0], self.action_dim[0], 1., 0.))
        for i in range(1, len(self.action_dim)):
            one_hot_action = tf.concat([one_hot_action, tf.one_hot(action[:, i], self.action_dim[i], 1., 0.)], axis=-1)

        embedded_action = self.action_embeddings(one_hot_action)
        rnn_state = self.flat_to_lstm_input(hidden_state)

        rnn_output, next_rnn_state = self.core(embedded_action, rnn_state)

        next_hidden_state = self.rnn_to_flat(next_rnn_state)

        # could add encoding but more research has to be done to why that is a good idea
        value_logits = self.value_head(next_hidden_state)
        value_logits = tf.nn.softmax(value_logits)
        value = self.value_encoder.decode(value_logits)

        # Rewards are only calculated in recurrent_inference.
        reward_logits = self.reward_head(rnn_output)
        reward_logits = tf.nn.softmax(reward_logits)
        reward = self.reward_encoder.decode(reward_logits)

        policy_logits = self.policy_head(next_hidden_state)

        outputs = {
            "value": value,
            "value_logits": value_logits,
            "reward": reward,
            "reward_logits": reward_logits,
            "policy_logits": policy_logits,
            "hidden_state": next_hidden_state
        }
        return outputs

    def encode_observation(self, observation):
        x = self.mlp1(observation)
        x = self.mlp2(x)
        # # Adding in the previous action and reward to the end of the LSTM output
        # # Can add in later if I feel like it
        # prev_stats = tf.keras.Input(shape=[sum(self.action_dim) + 1], dtype=np.float32)
        # x = tf.concat([x, prev_stats], axis=-1)

        return x

    def core(self, embedded_action, lstm_input):
        rnn_output, next_hidden_state = self._core(embedded_action, lstm_input)
        return rnn_output, next_hidden_state

    def to_hidden(self, encoded_observation):
        # mapping it to the size and domain of the hidden state
        x = self._to_hidden(encoded_observation)
        return x

    def value_head(self, x):
        x = self.head_hidden_layers(x)
        value_output = self._value_head(x)
        return value_output

    def reward_head(self, x):
        x = self.head_hidden_layers(x)
        reward_output = self._reward_head(x)
        return reward_output

    def policy_head(self, x):
        x = self.head_hidden_layers(x)
        shop_output = self._shop_output(x)
        item_output = self._item_output(x)
        bench_output = self._bench_output(x)
        board_output_x = self._board_output_x(x)
        board_output_y = self._board_output_y(x)
        return [shop_output, item_output, bench_output, board_output_x, board_output_y]

    def head_hidden_layers(self, x):
        def _make_layer(head_size):
            return [
                tf.keras.layers.Dense(head_size, use_bias=False),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.ReLU(),
            ]

        for idx, size in enumerate(self.head_hidden_sizes):
            x = tf.keras.Sequential(_make_layer(size), name='intermediate_{}'.format(idx))(x)

        return x

    def get_weights(self):
        # Returns the weights of this network.
        return []

    def training_steps(self) -> int:
        # How many steps / batches the network has been trained for.
        return 0

    # At the start of each search, we add dirichlet noise to the prior of the root
    # to encourage the search to explore new actions.
    def add_exploration_noise(self, node: Node):
        for act_dim in range(len(self.action_dim)):
            actions = list(node.children[act_dim].keys())
            noise = np.random.dirichlet([config.ROOT_DIRICHLET_ALPHA] * len(actions))
            frac = config.ROOT_EXPLORATION_FRACTION
            for a, n in zip(actions, noise):
                node.children[act_dim][a].prior = node.children[act_dim][a].prior * (1 - frac) + n * frac

    # Core Monte Carlo Tree Search algorithm.
    # To decide on an action, we run N simulations, always starting at the root of
    # the search tree and traversing the tree according to the UCB formula until we
    # reach a leaf node.
    def run_mcts(self, root: Node, action: List, player_num: int):
        min_max_stats = MinMaxStats(config.MINIMUM_REWARD, config.MAXIMUM_REWARD)

        for _ in range(config.NUM_SIMULATIONS):
            history = ActionHistory(action)
            node = root
            search_path = [node]

            # There is a chance I am supposed to check if the tree for the non-main-branch
            # Decision paths (axis 1-4) should be expanded. I am currently only expanding on the
            # main decision axis.
            while node.expanded():
                action, node = self.select_child(node, min_max_stats)
                history.add_action(action)
                search_path.append(node)

            # Inside the search tree we use the dynamics function to obtain the next
            # hidden state given an action and the previous hidden state.
            parent = search_path[-2]
            network_output = self.recurrent_inference(parent.hidden_state,
                                                      np.expand_dims(np.asarray(history.last_action()), axis=0))
            self.expand_node(node, self.player_to_play(player_num, history.last_action()), network_output)
            # print("value {}".format(network_output["value"]))
            self.backpropagate(search_path, network_output["value"], min_max_stats, player_num)

    # Select the child with the highest UCB score.
    def select_child(self, node: Node, min_max_stats: MinMaxStats):
        actions = []
        return_child = None
        for act_dim in range(len(self.action_dim)):
            _, action, child = max((self.ucb_score(node, child, min_max_stats), action,
                                    child) for action, child in node.children[act_dim].items())
            actions.append(action)
            if act_dim == 0:
                return_child = child
        if actions[0] == 7:
            node.to_play *= -1
        return actions, return_child

    # The score for a node is based on its value, plus an exploration bonus based on
    # the prior.
    def ucb_score(self, parent: Node, child: Node, min_max_stats: MinMaxStats) -> float:
        pb_c = math.log((parent.visit_count + config.PB_C_BASE + 1) /
                        config.PB_C_BASE) + config.PB_C_INIT
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior
        value_score = min_max_stats.normalize(child.value())
        return prior_score + value_score

    def player_to_play(self, player_num, action):
        if action == 7:
            if player_num == config.NUM_PLAYERS - 1:
                return 0
            else:
                return player_num + 1
        return player_num

    def increment_player(self, player_num):
        if player_num < config.NUM_PLAYERS - 1:
            return player_num + 1
        return 0

    # At the end of a simulation, we propagate the evaluation all the way up the
    # tree to the root.
    def backpropagate(self, search_path: List[Node], value: float,
                      min_max_stats: MinMaxStats, player_num: int):
        for node in search_path:
            node.value_sum += value if node.to_play == player_num else -value
            node.visit_count += 1
            min_max_stats.update(node.value())

            value = node.reward + config.DISCOUNT * value

    def select_action(self, node: Node):
        action = []
        for act_dim in range(len(self.action_dim)):
            visit_counts = [
                (child.visit_count, action) for action, child in node.children[act_dim].items()
            ]
            t = self.visit_softmax_temperature()
            action.append(histogram_sample(visit_counts, t, use_softmax=False))
        return action

    def visit_softmax_temperature(self):
        return .1
        # if num_moves < 30:
        #     return 0.1
        # else:
        #     return 0.0  # Play according to the max.

    def flat_to_lstm_input(self, state):
        """Maps flat vector to LSTM state."""
        tensors = []
        cur_idx = 0
        for size in [config.HIDDEN_STATE_SIZE]:
            states = (state[Ellipsis, cur_idx:cur_idx + size],
                      state[Ellipsis, cur_idx:cur_idx + size])
            cur_idx += 2 * size
            tensors.append(states)
            cur_idx = 0
        # assert cur_idx == state.shape[-1]
        return tensors

    def rnn_to_flat(self, state):
        """Maps LSTM state to flat vector."""
        states = []
        for cell_state in state:
            if not (isinstance(cell_state, list) or isinstance(cell_state, tuple)):
                # This is a GRU or SimpleRNNCell
                cell_state = (cell_state,)
            states.extend(cell_state)
        return tf.concat(states, -1)

    def action_embeddings(self, action):
        x = tf.keras.layers.Dense(config.HIDDEN_STATE_SIZE)(action)
        return x

    #### This code is not used anywhere. Leaving here in case people want to play around with masking the input ###
    #### So that training does not see any values that are not at least attempted to be executed. ####
    def maskInput(self, logits, actions):
        if actions[0] < 8 or actions[0] > 8:
            actions[1] = self.action_dim[1] - 1
            one_hot_action = tf.expand_dims(tf.squeeze(tf.one_hot(actions[1], self.action_dim[1])), axis=0)
            logits[1] = one_hot_action
        elif actions[1] == self.action_dim[1] - 1:
            actions[1] = np.random.randint(self.action_dim[1] - 1)
        if actions[0] < 9 or actions[0] == 11:
            actions[2] = self.action_dim[2] - 1
            one_hot_action = tf.expand_dims(tf.squeeze(tf.one_hot(actions[2], self.action_dim[2])), axis=0)
            logits[2] = one_hot_action
        elif actions[2] == self.action_dim[2] - 1:
            actions[2] = np.random.randint(self.action_dim[2] - 1)
        if actions[0] < 8 or actions[0] == 9:
            actions[3] = self.action_dim[3] - 1
            one_hot_action = tf.expand_dims(tf.squeeze(tf.one_hot(actions[3], self.action_dim[3])), axis=0)
            logits[3] = one_hot_action
            actions[4] = self.action_dim[4] - 1
            one_hot_action = tf.expand_dims(tf.squeeze(tf.one_hot(actions[4], self.action_dim[4])), axis=0)
            logits[4] = one_hot_action
        else:
            if actions[3] == self.action_dim[3] - 1:
                actions[3] = np.random.randint(self.action_dim[3] - 1)
            if actions[4] == self.action_dim[4] - 1:
                actions[4] = np.random.randint(self.action_dim[4] - 1)
        return logits, actions

    def action_history(self):
        return ActionHistory(self.history)

    def get_rl_training_variables(self):
        return self.trainable_variables


class Mlp(tf.Module):
    def __init__(self, hidden_size=256, mlp_dim=512):
        super(Mlp, self).__init__()
        self.fc1 = tf.keras.layers.Dense(mlp_dim, dtype=tf.float32)
        self.fc2 = tf.keras.layers.Dense(hidden_size, dtype=tf.float32)
        self.norm = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm2(x)
        x = tf.keras.activations.relu(x)
        x = self.fc2(x)
        x = self.norm(x)
        x = tf.keras.activations.relu(x)
        return x

    def __call__(self, x, *args, **kwargs):
        out = self.forward(x)
        return out


class ActionHistory(object):
    """Simple history container used inside the search.

  Only used to keep track of the actions executed.
  """

    def __init__(self, history: List):
        self.history = [history]

    def clone(self):
        return ActionHistory(self.history)

    def add_action(self, action: List):
        self.history.append(action)

    def last_action(self) -> List:
        return self.history[-1]


def masked_distribution(x, use_exp, mask=None):
    if mask is None:
        mask = [1] * len(x)
    assert sum(mask) > 0, 'Not all values can be masked.'
    assert len(mask) == len(x), (
        'The dimensions of the mask and x need to be the same.')
    x = np.exp(x) if use_exp else np.array(x, dtype=np.float64)
    mask = np.array(mask, dtype=np.float64)
    x *= mask
    if sum(x) == 0:
        # No unmasked value has any weight. Use uniform distribution over unmasked
        # tokens.
        x = mask
    return x / np.sum(x, keepdims=True)


def masked_softmax(x, mask=None):
    x = np.array(x) - np.max(x, axis=-1)  # to avoid overflow
    return masked_distribution(x, use_exp=True, mask=mask)


def masked_count_distribution(x, mask=None):
    return masked_distribution(x, use_exp=False, mask=mask)


def histogram_sample(distribution, temperature, use_softmax=False, mask=None):
    actions = [d[1] for d in distribution]
    visit_counts = np.array([d[0] for d in distribution], dtype=np.float64)
    if temperature == 0.:
        probs = masked_count_distribution(visit_counts, mask=mask)
        return actions[np.argmax(probs)]
    if use_softmax:
        logits = visit_counts / temperature
        probs = masked_softmax(logits, mask)
    else:
        logits = visit_counts ** (1. / temperature)
        probs = masked_count_distribution(logits, mask)
    return np.random.choice(actions, p=probs)


class ValueEncoder:
    """Encoder for reward and value targets from Appendix of MuZero Paper."""

    def __init__(self,
                 min_value,
                 max_value,
                 num_steps,
                 use_contractive_mapping=True):
        if not max_value > min_value:
            raise ValueError('max_value must be > min_value')
        # min_value = float(min_value)
        # max_value = float(max_value)
        if use_contractive_mapping:
            max_value = contractive_mapping(max_value)
            min_value = contractive_mapping(min_value)
        if num_steps <= 0:
            num_steps = int(math.ceil(max_value) + 1 - math.floor(min_value))
        self.min_value = min_value
        self.max_value = max_value
        self.value_range = max_value - min_value
        self.num_steps = num_steps
        self.step_size = self.value_range / (num_steps - 1)
        self.step_range_int = tf.range(self.num_steps, dtype=tf.int32)
        self.step_range_float = tf.cast(self.step_range_int, tf.float32)
        self.use_contractive_mapping = use_contractive_mapping

    def encode(self, value):
        if len(value.shape) != 1:
            raise ValueError(
                'Expected value to be 1D Tensor [batch_size], but got {}.'.format(
                    value.shape))
        if self.use_contractive_mapping:
            value = contractive_mapping(value)
        value = tf.expand_dims(value, -1)
        clipped_value = tf.clip_by_value(value, self.min_value, self.max_value)
        above_min = clipped_value - self.min_value
        num_steps = above_min / self.step_size
        lower_step = tf.math.floor(num_steps)
        upper_mod = num_steps - lower_step
        lower_step = tf.cast(lower_step, tf.int32)
        upper_step = lower_step + 1
        lower_mod = 1.0 - upper_mod
        lower_encoding, upper_encoding = (
            tf.cast(tf.math.equal(step, self.step_range_int), tf.float32) * mod
            for step, mod in (
                (lower_step, lower_mod),
                (upper_step, upper_mod),
            ))
        return lower_encoding + upper_encoding

    def decode(self, logits):
        if len(logits.shape) != 2:
            raise ValueError(
                'Expected logits to be 2D Tensor [batch_size, steps], but got {}.'
                .format(logits.shape))
        num_steps = tf.reduce_sum(logits * self.step_range_float, -1)
        above_min = num_steps * self.step_size
        value = above_min + self.min_value
        if self.use_contractive_mapping:
            value = inverse_contractive_mapping(value)
        return value


# From the MuZero paper.
def contractive_mapping(x, eps=0.001):
    return tf.math.sign(x) * (tf.math.sqrt(tf.math.abs(x) + 1.) - 1.) + eps * x


# From the MuZero paper.
def inverse_contractive_mapping(x, eps=0.001):
    return tf.math.sign(x) * (
            tf.math.square(
                (tf.sqrt(4 * eps *
                         (tf.math.abs(x) + 1. + eps) + 1.) - 1.) / (2. * eps)) - 1.)

# Possibilities for the other 4 dimensions on the action space
# If I get an option that requires one of my other dimensions to be non-standard
# I can either take a maximum as I would do in the normal setting or I can create
# Another tree for the separate dimensions and search down that tree for the right option
# I would only use the smaller dimension on the root and then use the main option
# For all of the other nodes. I think this is what should be implemented in the long run
# But for now to get the algorithm running, implementing a simple max across the space is
# Fine.
