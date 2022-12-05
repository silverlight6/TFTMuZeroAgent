from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Dict, List
from numba import jit, cuda, typed

import collections
import math
import config
import numpy as np
import tensorflow as tf
import time

##########################
####### Helpers ##########

MAXIMUM_FLOAT_VALUE = float('inf')

KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])

NetworkOutput = collections.namedtuple(
    'NetworkOutput',
    'value reward policy_logits hidden_state')
##### JITTED FUNCTIONS ######
@jit(target_backend='cuda', nopython = True)
def normalize2(value, min, max):
    ans = (value - min)/(max-min)
    return ans

class MinMaxStats(object):
    """A class that holds the min-max values of the tree."""

    def __init__(self, minimum: int, maximum: int):
        self.maximum = minimum
        self.minimum = maximum
        self.set = False

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)
        self.set = True 

    def normalize(self, value: float) -> float:

        #if self.maximum > self.minimum:
        if self.set == True:
            #print("Time = ", time.time_ns() - ckpt)
            
            # We normalize only when we have set the maximum and minimum values.
            
            if value == 0:
                value = 0.0
            max =self.maximum
            min = self.minimum
            #print("val, max, min = ", value, max, min)
            #print("types:", type(value), type(max), type(min))
            if type(max) != int and type(max) != float:
                max = float(max.numpy())
            if type(min) != int and type(min) != float:
                min = float(min.numpy())
            if type(value) != int and type(value) != float:
                value = float(value.numpy())

            ans = normalize2(value, max, min)
            #print("val, max, min = ", value, max, min)

            return ans

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


class Network(tf.Module):
    """
    Base class for all of MuZero neural networks.
    """

    def __init__(self,
                 representation: tf.keras.Model,
                 dynamics: tf.keras.Model,
                 prediction: tf.keras.Model
                 ) -> None:
        super().__init__(name='MuZeroAgent')
        self.config = config
        self.representation: tf.keras.Model = representation
        self.dynamics: tf.keras.Model = dynamics
        self.prediction: tf.keras.Model = prediction
        # print(dynamics.summary())
        # print(prediction.summary())

        self.value_encoder = ValueEncoder(*tuple(map(inverse_contractive_mapping, (-300., 300.))), 0)

        self.reward_encoder = ValueEncoder(*tuple(map(inverse_contractive_mapping, (-300., 300.))), 0)

        self.initial_inference_model: tf.keras.Model = self.build_initial_inference_model()
        self.recurrent_inference_model: tf.keras.Model = self.build_recurrent_inference_model()

        self.ckpt_time = time.time_ns()

    def build_initial_inference_model(self) -> tf.keras.Model:
        observation = tf.keras.Input(shape=config.INPUT_SHAPE, dtype=tf.float32, name='observation')

        hidden_state = self.representation(observation)

        value, policy_logits = self.prediction(hidden_state)

        return tf.keras.Model(inputs=observation,
                              outputs=[hidden_state, value, policy_logits],
                              name='initial_inference')

    def initial_inference(self, observation) -> dict:
        # self.ckpt_time = time.time_ns()
        hidden_state, value_logits, policy_logits = \
            self.initial_inference_model(observation, training=False)
        # print("initial_inference takes {} time".format(time.time_ns() - self.ckpt_time))
        # self.ckpt_time = time.time_ns()
        value = self.value_encoder.decode(value_logits)
        reward = tf.zeros_like(value)
        reward_logits = self.reward_encoder.encode(reward)
        # print("initial_inference value / reward takes {} time".format(time.time_ns() - self.ckpt_time))

        outputs = {
            "value": value,
            "value_logits": value_logits,
            "reward": reward,
            "reward_logits": reward_logits,
            "policy_logits": policy_logits,
            "hidden_state": hidden_state
        }
        return outputs

    def build_recurrent_inference_model(self) -> tf.keras.Model:
        hidden_state = tf.keras.Input(shape=[2 * config.HIDDEN_STATE_SIZE], dtype=tf.float32, name='hidden_state')
        # state_space = tf.keras.Input(shape=[1, config.HIDDEN_STATE_SIZE], dtype=tf.float32, name='state_space')
        action = tf.keras.Input(shape=([config.ACTION_CONCAT_SIZE]), dtype=tf.int32, name='action')

        new_hidden_state, reward = self.dynamics((hidden_state, action))

        value, policy_logits = self.prediction(new_hidden_state)

        return tf.keras.Model(inputs=[hidden_state, action],
                              outputs=[new_hidden_state, reward, value, policy_logits],
                              name='recurrent_inference')

    def recurrent_inference(self, hidden_state, action) -> dict:
        one_hot_action = tf.Variable(tf.one_hot(action[:, 0], config.ACTION_DIM[0], 1., 0.))
        for i in range(1, len(config.ACTION_DIM)):
            one_hot_action = tf.concat([one_hot_action, tf.one_hot(action[:, i], config.ACTION_DIM[i], 1., 0.)],
                                       axis=-1)

        hidden_state, reward_logits, value_logits, policy_logits = \
            self.recurrent_inference_model((hidden_state, one_hot_action), training=False)

        value = self.value_encoder.decode(value_logits)
        reward = self.reward_encoder.decode(reward_logits)

        outputs = {
            "value": value,
            "value_logits": value_logits,
            "reward": reward,
            "reward_logits": reward_logits,
            "policy_logits": policy_logits,
            "hidden_state": hidden_state
        }
        return outputs

    def rnn_to_flat(self, state):
        """Maps LSTM state to flat vector."""
        states = []
        for cell_state in state:
            states.extend(cell_state)
        return tf.concat(states, -1, name="flat_concat")

    @staticmethod
    def flat_to_lstm_input(state):
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


class TFTNetwork(Network):
    """
    Neural networks for tic-tac-toe game.
    """

    def __init__(self) -> None:
        regularizer = tf.keras.regularizers.l2(l=1e-4)

        # Representation model. Observation --> hidden state
        representation_model: tf.keras.Model = tf.keras.Sequential([
            tf.keras.Input(shape=config.INPUT_SHAPE),
            Mlp(hidden_size=config.HIDDEN_STATE_SIZE, mlp_dim=config.HEAD_HIDDEN_SIZE),
            tf.keras.layers.Dense(2 * config.HIDDEN_STATE_SIZE, activation='sigmoid', name='final'),
            tf.keras.layers.Flatten()
        ], name='observation_encoding')

        # Dynamics Model. Hidden State --> next hidden state and reward
        # Action encoding
        encoded_state_action = tf.keras.Input(shape=[config.ACTION_CONCAT_SIZE])
        action_embeddings = tf.keras.layers.Dense(units=2 * config.HIDDEN_STATE_SIZE, activation='relu',
                                                  kernel_regularizer=regularizer,
                                                  bias_regularizer=regularizer)(encoded_state_action)
        action_embeddings = tf.keras.layers.Flatten()(action_embeddings)

        # Hidden state input. [[1, 256], [1. 256]] Needs both the hidden state and lstm state
        dynamic_hidden_state = tf.keras.Input(shape=[2 * config.HIDDEN_STATE_SIZE], name='hidden_state_input')
        # dynamic_state_space = tf.keras.Input(shape=[config.HIDDEN_STATE_SIZE], name='lstm_state_space')
        # dynamic_hidden_state = [[dynamic_hidden_state[0]], [dynamic_hidden_state[1]]]
        rnn_state = self.flat_to_lstm_input(dynamic_hidden_state)

        # Core of the model
        rnn_cell_cls = {
            'lstm': tf.keras.layers.LSTMCell,
        }['lstm']
        rnn_cells = [
            rnn_cell_cls(
                config.HIDDEN_STATE_SIZE,
                recurrent_activation='sigmoid',
                name='cell_{}'.format(idx)) for idx in range(1)]
        core = tf.keras.layers.StackedRNNCells(rnn_cells, name='recurrent_core')

        rnn_output, next_rnn_state = core(action_embeddings, rnn_state)
        next_hidden_state = self.rnn_to_flat(next_rnn_state)

        # Reward head
        x = tf.keras.layers.Dense(units=601, activation='tanh', name='reward',
                                  kernel_regularizer=regularizer, bias_regularizer=regularizer)(rnn_output)
        reward_output = tf.keras.layers.Softmax()(x)
        dynamics_model: tf.keras.Model = \
            tf.keras.Model(inputs=[dynamic_hidden_state, encoded_state_action],
                           outputs=[next_hidden_state, reward_output], name='dynamics')

        pred_hidden_state = tf.keras.Input(shape=np.array([2 * config.HIDDEN_STATE_SIZE]), name="prediction_input")
        x = Mlp(hidden_size=config.HIDDEN_STATE_SIZE, mlp_dim=config.HEAD_HIDDEN_SIZE)(pred_hidden_state)
        x = tf.keras.layers.Dense(units=601, activation='tanh', name='value',
                                  kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
        value_output = tf.keras.layers.Softmax()(x)
        shop_output = tf.keras.layers.Dense(config.ACTION_DIM[0], activation='softmax', name='shop_layer')(x)
        item_output = tf.keras.layers.Dense(config.ACTION_DIM[1], activation='softmax', name='item_layer')(x)
        bench_output = tf.keras.layers.Dense(config.ACTION_DIM[2], activation='softmax', name='bench_layer')(x)
        board_output_x = tf.keras.layers.Dense(config.ACTION_DIM[3], activation='softmax', name='board_x_layer')(x)
        board_output_y = tf.keras.layers.Dense(config.ACTION_DIM[4], activation='softmax', name='board_y_layer')(x)

        prediction_model: tf.keras.Model = tf.keras.Model(inputs=pred_hidden_state,
                                                          outputs=[value_output,
                                                                   [shop_output, item_output, bench_output,
                                                                    board_output_x, board_output_y]],
                                                          name='prediction')

        super().__init__(representation=representation_model,
                         dynamics=dynamics_model,
                         prediction=prediction_model)


class Mlp(tf.Module):
    def __init__(self, hidden_size=256, mlp_dim=512):
        super(Mlp, self).__init__()
        self.fc1 = tf.keras.layers.Dense(mlp_dim, dtype=tf.float32)
        self.fc2 = tf.keras.layers.Dense(hidden_size, dtype=tf.float32)
        self.norm = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = tf.keras.activations.relu(x)
        x = self.fc2(x)
        x = self.norm2(x)
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

##### JITTED FUNCTIONS #######
@jit(target_backend='cuda', nopython=True)
def expand_node2(network_output, action_dim):
    policy = []
    for i, action_dim in enumerate(action_dim):
        policy.append({b: math.exp(network_output[i][0][b]) for b in range(action_dim)})
    return policy

@jit(target_backend='cuda', nopython = True)
def ucb_score_2(parent_visit_count,child_visit_count, PB_C_BASE, PB_C_INIT,prior):
    pb_c = math.log((parent_visit_count + PB_C_BASE + 1) /
                        PB_C_BASE) + PB_C_INIT
    pb_c *= math.sqrt(parent_visit_count) / (child_visit_count + 1)

    prior_score = pb_c * prior
    return prior_score
# @jit(target_backend='cuda', nopython = True)
# def select_child2(items, ucb):
#     _, action, child = max((ucb, action,child) for action, child in items)
#     return(_,action,child)
class MCTSAgent:
    """
    Use Monte-Carlo Tree-Search to select moves.
    """

    def __init__(self,
                 network: Network,
                 agent_id: int
                 ) -> None:
        self.network: Network = network
        self.agent_id = agent_id

        # action_dim = [possible actions, item bench, unit bench, x axis, y axis]
        self.action_dim = [12, 10, 9, 7, 4]
        self.num_actions = 0
        self.ckpt_time = time.time_ns()

    def expand_node(self, node: Node, to_play: int, network_output):
        ckpt = time.time_ns() 
        node.to_play = to_play
        node.hidden_state = network_output["hidden_state"]
        node.reward = network_output["reward"]

        input_to_jitfunc = [] 
        for i in network_output["policy_logits"]:
            input_to_jitfunc.append(i.numpy())
        # print(input_to_jitfunc)
        # print("^ input")
        # print(network_output["policy_logits"])
        # print("^ logits")
        # print(self.action_dim)
        # print("^ act dim")
        try:
            policy = expand_node2(input_to_jitfunc, [12,10,9,7,4]) #hardcoding in self.action_dim because it changes type randomly. 
            for i, action_dim in enumerate(policy):
                policy_sum = sum(action_dim.values())
                for action, p in action_dim.items():
                    node.children[i][action] = Node(p / policy_sum)
        except:
            print("error - reverting to old function")
            self.expand_node_old(node, network_output)
        
        if np.random.randint(0,1000) == 500:
            print("expand_node took {} time".format(time.time_ns() - ckpt))
    def expand_node_old(self, node: Node, network_output):
        policy = []
        for i, action_dim in enumerate(self.action_dim):
            policy.append({b: math.exp(network_output["policy_logits"][i][0][b]) for b in range(action_dim)})
        for i, action_dim in enumerate(policy):
            policy_sum = sum(action_dim.values())
            for action, p in action_dim.items():
                node.children[i][action] = Node(p / policy_sum)
    def add_exploration_noise(self, node: Node):
        for act_dim in range(len(self.action_dim)):
            actions = list(node.children[act_dim].keys())
            noise = np.random.dirichlet([config.ROOT_DIRICHLET_ALPHA] * len(actions))
            frac = config.ROOT_EXPLORATION_FRACTION
            for a, n in zip(actions, noise):
                node.children[act_dim][a].prior = node.children[act_dim][a].prior * (1 - frac) + n * frac

    # Select the child with the highest UCB score.
    def select_child(self, node: Node, min_max_stats: MinMaxStats):
        ckpt = time.time_ns()
        actions = []
        return_child = None
        for act_dim in range(len(self.action_dim)):
            # items = node.children[act_dim].items()
            # ucb = self.ucb_score(node, child, min_max_stats)

            _, action, child = max((self.ucb_score(node, child, min_max_stats), action,
                                    child) for action, child in node.children[act_dim].items())
            actions.append(action)
            if act_dim == 0:
                return_child = child
        if actions[0] == 7:
            node.to_play *= -1
        
        if np.random.randint(0,1000) == 500:
            print("select child took {} time".format(time.time_ns() - ckpt))
        return actions, return_child

    # The score for a node is based on its value, plus an exploration bonus based on
    # the prior.
    @staticmethod
    def ucb_score(parent: Node, child: Node, min_max_stats: MinMaxStats) -> float:
        ckpt = time.time_ns()
        pb_c = math.log((parent.visit_count + config.PB_C_BASE + 1) /
                         config.PB_C_BASE) + config.PB_C_INIT
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
        prior_score = pb_c*child.prior
        #prior_score = ucb_score_2(parent_visit_count, child_visit_count, PB_C_BASE, PB_C_INIT, prior)
        #print("ucb took {} time".format(time.time_ns() - ckpt))
        value_score = min_max_stats.normalize(child.value())
        #print("normalize took {} time".format(time.time_ns() - ckpt))

        return prior_score + value_score

    # At the end of a simulation, we propagate the evaluation all the way up the
    # tree to the root.
    @staticmethod
    def backpropagate(search_path: List[Node], value: float,
                      min_max_stats: MinMaxStats, player_num: int):
        for node in search_path:
            node.value_sum += value if node.to_play == player_num else -value
            node.visit_count += 1
            min_max_stats.update(node.value())

            value = node.reward + config.DISCOUNT * value

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
            self.ckpt_time = time.time_ns()
            while node.expanded():
                action, node = self.select_child(node, min_max_stats)
                history.add_action(action)
                search_path.append(node)

            # Inside the search tree we use the dynamics function to obtain the next
            # hidden state given an action and the previous hidden state.
            parent = search_path[-2]
            #print("mcts child select takes {} time".format(time.time_ns() - self.ckpt_time))

            network_output = self.network.\
                recurrent_inference(parent.hidden_state, np.expand_dims(np.asarray(history.last_action()), axis=0))
            self.ckpt_time = time.time_ns()
            self.expand_node(node, self.player_to_play(player_num, history.last_action()), network_output)
            # print("value {}".format(network_output["value"]))
            self.backpropagate(search_path, network_output["value"], min_max_stats, player_num)
            #print("mcts expand/backprop takes {} time".format(time.time_ns() - self.ckpt_time))

    def select_action(self, node: Node):
        action = []
        for act_dim in range(len(self.action_dim)):
            visit_counts = [
                (child.visit_count, action) for action, child in node.children[act_dim].items()
            ]
            t = self.visit_softmax_temperature()
            action.append(self.histogram_sample(visit_counts, t, use_softmax=False))
        return action

    def policy(self, observation, player_num):
        root = Node(0)

        network_output = self.network.initial_inference(observation)
        self.expand_node(root, player_num, network_output)
        self.add_exploration_noise(root)

        initial_action = [
            np.argmax(network_output["policy_logits"][0].numpy()),
            np.argmax(network_output["policy_logits"][1].numpy()),
            np.argmax(network_output["policy_logits"][2].numpy()),
            np.argmax(network_output["policy_logits"][3].numpy()),
            np.argmax(network_output["policy_logits"][4].numpy())
        ]

        self.run_mcts(root, initial_action, player_num)

        self.ckpt_time = time.time_ns()
        action = self.select_action(root)
        # print("selecting actions takes {} time".format(time.time_ns() - self.ckpt_time))

        # Masking only if training is based on the actions taken in the environment.
        # Training in MuZero is mostly based on the predicted actions rather than the real ones.
        # network_output["policy_logits"], action = self.maskInput(network_output["policy_logits"], action)

        # Notes on possibilities for other dimensions at the bottom
        self.num_actions += 1

        return action, network_output["policy_logits"]

    def fill_metadata(self) -> Dict[str, str]:
        return {'network_id': str(self.network.training_steps())}

    @staticmethod
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

    @staticmethod
    def player_to_play(player_num, action):
        if action == 7:
            if player_num == config.NUM_PLAYERS - 1:
                return 0
            else:
                return player_num + 1
        return player_num

    @staticmethod
    def visit_softmax_temperature():
        return .1


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
