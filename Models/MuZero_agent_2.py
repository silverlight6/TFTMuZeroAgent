from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Dict, List

import collections
import math
import config
import numpy as np
import tensorflow as tf
import time
import Simulator.utils as utils
from Simulator.stats import COST

##########################
####### Helpers ##########

MAXIMUM_FLOAT_VALUE = float('inf')

KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])

NetworkOutput = collections.namedtuple(
    'NetworkOutput',
    'value reward policy_logits hidden_state')


def normalize2(value, minimum, maximum):  # faster implementation of normalization
    if minimum != maximum:
        ans = (value - minimum) / (maximum - minimum)
        return ans
    else:
        return maximum


def update2(value, maximum, minimum):
    if value > maximum:
        maximum = value 
    elif value < minimum:
        minimum = value
    return maximum, minimum


class MinMaxStats(object):
    """A class that holds the min-max values of the tree."""

    def __init__(self, minimum: float, maximum: float):
        self.maximum = minimum
        self.minimum = maximum
        self.set = False

    def update(self, value: float):
        self.maximum, self.minimum = update2(float(value), float(self.maximum), float(self.minimum))
        self.set = True       

    def normalize(self, value: float) -> float:

        if self.set is True:

            # We normalize only when we have set the maximum and minimum values (signified by self.set = True).
            if value == 0:
                value = 0.0
            maximum = self.maximum
            minimum = self.minimum
            # Convert from 1d tensor to float if needed (JIT function requires float)
            if type(maximum) != int and type(maximum) != float:
                maximum = float(maximum)
            if type(minimum) != int and type(minimum) != float:
                minimum = float(minimum)
            if type(value) != int and type(value) != float:
                value = float(value)
            ans = normalize2(value, maximum, minimum)
            return ans

        return value


class Node(object):

    # initialize a new node with given prior probability
    def __init__(self, prior: float):
        self.visit_count = 0
        self.to_play = 1
        self.prior = prior
        self.value_sum = 0
        # Initialize empty dictionart to store children nodes
        # 5 because there are 5 separate actions.
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    # check if the node has been expanded (i.e has children)
    def expanded(self) -> bool:
        return len(self.children) > 0

    # calculate the value of the node as an average of visited nodes.
    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


class Network(tf.keras.Model):
    """
    Base class for all of MuZero neural networks.
    """
    # initialize the network with the given representation, dynamics, and prediction model.
    def __init__(self,
                 representation: tf.keras.Model,
                 dynamics: tf.keras.Model,
                 prediction: tf.keras.Model
                 ) -> None:
        super().__init__(name='MuZeroAgent')
        # temp
        self.rec_count = 0

        self.config = config
        self.representation: tf.keras.Model = representation
        self.dynamics: tf.keras.Model = dynamics
        self.prediction: tf.keras.Model = prediction

        # create encoders for the value and reward, in order to put them in a form suitable for training.
        self.value_encoder = ValueEncoder(*tuple(map(inverse_contractive_mapping, (-300., 300.))), 0)

        self.reward_encoder = ValueEncoder(*tuple(map(inverse_contractive_mapping, (-300., 300.))), 0)

        # build initial and recurrent inference models.
        self.initial_inference_model: tf.keras.Model = self.build_initial_inference_model()
        self.recurrent_inference_model: tf.keras.Model = self.build_recurrent_inference_model()

        self.ckpt_time = time.time_ns()

    # build the initial inference model (used to generate predicitons)
    def build_initial_inference_model(self) -> tf.keras.Model:
        # define the input tensor
        observation = tf.keras.Input(shape=config.INPUT_SHAPE, dtype=tf.float32, name='observation')

        hidden_state = self.representation(observation)

        value, policy_logits = self.prediction(hidden_state)

        return tf.keras.Model(inputs=observation,
                              outputs=[hidden_state, value, policy_logits],
                              name='initial_inference')

    # Apply the initial inference model to the given hidden state
    def initial_inference(self, observation, training=False) -> dict:
        hidden_state, value_logits, policy_logits = \
            self.initial_inference_model(observation, training=training)
        value = self.value_encoder.decode(tf.nn.softmax(value_logits))
        reward = tf.zeros_like(value)
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

    # Build the recurrent inference model (used to generate predictions for subsequent steps)
    def build_recurrent_inference_model(self) -> tf.keras.Model:
        hidden_state = tf.keras.Input(shape=[2 * config.HIDDEN_STATE_SIZE], dtype=tf.float32, name='hidden_state')
        # state_space = tf.keras.Input(shape=[1, config.HIDDEN_STATE_SIZE], dtype=tf.float32, name='state_space')
        action = tf.keras.Input(shape=([config.ACTION_CONCAT_SIZE]), dtype=tf.int32, name='action')

        new_hidden_state, reward = self.dynamics((hidden_state, action))

        value, policy_logits = self.prediction(new_hidden_state)

        return tf.keras.Model(inputs=[hidden_state, action],
                              outputs=[new_hidden_state, reward, value, policy_logits],
                              name='recurrent_inference')

    def decode_action(self, str_action):
        num_items = str_action.count("_")
        split_action = str_action.split("_")
        element_list = [0,0,0]
        for i in range(num_items+1):
            element_list[i] = int(split_action[i])
        return np.asarray(element_list)

    # Apply the recurrent inference model to the given hidden state
    def recurrent_inference(self, hidden_state, action, training=False) -> dict:  # CHECKPOINT
        action_list = [self.decode_action(action[i]) for i in range(len(action))]
        action_batch = np.asarray(action_list)

        one_hot_action = tf.one_hot(action_batch[:, 0], config.ACTION_DIM[0], 1., 0., axis=-1)
        one_hot_target_a = tf.one_hot(action_batch[:, 1], config.ACTION_DIM[1], 1., 0., axis=-1)
        one_hot_target_b = tf.one_hot(action_batch[:, 2], config.ACTION_DIM[1] - 1, 1., 0., axis=-1)

        final_action = tf.concat([one_hot_action, one_hot_target_a, one_hot_target_b], axis=-1)
        
        hidden_state, reward_logits, value_logits, policy_logits = \
            self.recurrent_inference_model((hidden_state, final_action), training=training)

        value = self.value_encoder.decode(tf.nn.softmax(value_logits))
        reward = self.reward_encoder.decode(tf.nn.softmax(reward_logits))

        outputs = {
            "value": value,
            "value_logits": value_logits,
            "reward": reward,
            "reward_logits": reward_logits,
            "policy_logits": policy_logits,
            "hidden_state": hidden_state
        }
        self.rec_count += 1 
        
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
        reward_output = tf.keras.layers.Dense(units=601, name='reward', kernel_regularizer=regularizer,
                                              bias_regularizer=regularizer)(rnn_output)

        dynamics_model: tf.keras.Model = \
            tf.keras.Model(inputs=[dynamic_hidden_state, encoded_state_action],
                           outputs=[next_hidden_state, reward_output], name='dynamics')

        pred_hidden_state = tf.keras.Input(shape=np.array([2 * config.HIDDEN_STATE_SIZE]), name="prediction_input")
        x = Mlp(hidden_size=config.HIDDEN_STATE_SIZE, mlp_dim=config.HEAD_HIDDEN_SIZE)(pred_hidden_state)
        value_x = Mlp(hidden_size=config.HIDDEN_STATE_SIZE, mlp_dim=config.HEAD_HIDDEN_SIZE)(x)
        value_output = tf.keras.layers.Dense(units=601, name='value', kernel_regularizer=regularizer,
                                             bias_regularizer=regularizer)(value_x)

        policy_x = Mlp(hidden_size=config.HIDDEN_STATE_SIZE, mlp_dim=config.HEAD_HIDDEN_SIZE)(x)
        policy_output_action = tf.keras.layers.Dense(config.ACTION_DIM[0], activation='relu', name='action_layer')(policy_x)
        policy_output_target = tf.keras.layers.Dense(config.ACTION_DIM[1], activation='relu', name='target_layer')(policy_x)
        policy_output_item = tf.keras.layers.Dense(config.ACTION_DIM[2], activation='relu', name='item_layer')(policy_x)

        prediction_model: tf.keras.Model = tf.keras.Model(inputs=pred_hidden_state,
                                                          outputs=[value_output, 
                                                          [policy_output_action, policy_output_target, policy_output_item]],
                                                          name='prediction')

        super().__init__(representation=representation_model,
                         dynamics=dynamics_model,
                         prediction=prediction_model)

    # Renaming as to not override built-in functions
    def tft_save_model(self, episode):
        self.save_weights("./Checkpoints/checkpoint_{}".format(episode))

    # Renaming as to not override built-in functions
    def tft_load_model(self, episode):
        self.load_weights("./Checkpoints/checkpoint_{}".format(episode))
        print("Loading model episode {}".format(episode))

    def get_rl_training_variables(self):
        return self.trainable_variables


class Mlp(tf.keras.Model):
    def __init__(self, hidden_size=256, mlp_dim=512):
        super(Mlp, self).__init__()
        self.fc1 = tf.keras.layers.Dense(mlp_dim, dtype=tf.float32)
        self.fc2 = tf.keras.layers.Dense(hidden_size, dtype=tf.float32)
        self.norm = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = tf.keras.activations.elu(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = tf.keras.activations.elu(x)
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
        min_value = float(min_value)
        max_value = float(max_value)
        if use_contractive_mapping:
            max_value = contractive_mapping(max_value)
            min_value = contractive_mapping(min_value)
        if num_steps <= 0:
            num_steps = tf.math.ceil(max_value) + 1 - tf.math.floor(min_value)
        self.min_value = min_value
        self.max_value = max_value
        self.value_range = max_value - min_value
        self.num_steps = num_steps
        self.step_size = self.value_range / (num_steps - 1)
        self.step_range_int = tf.range(self.num_steps, dtype=tf.int32)
        self.step_range_float = tf.cast(self.step_range_int, tf.float32)
        self.use_contractive_mapping = use_contractive_mapping

    def encode(self, value):  # not worth optimizing
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
                (upper_step, upper_mod),)
        )
        return lower_encoding + upper_encoding

    def decode(self, logits):  # not worth optimizing
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
    return tf.math.sign(x) * \
           (tf.math.square((tf.sqrt(4 * eps * (tf.math.abs(x) + 1. + eps) + 1.) - 1.) / (2. * eps)) - 1.)


def expand_node2(network_output, action_dim):
    policy = [{b: math.exp(network_output[b]) for b in range(action_dim)}]
    return policy


# EXPLANATION OF MCTS:
"""
1. select leaf node with maximum value using method called UCB1 
2. expand the leaf node, adding children for each possible action
3. Update leaf node and ancestor values using the values learnt from the children
 - values for the children are generated using neural network 
4. Repeat above steps a given number of times
5. Select path with highest value
"""


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

        self.action_dim = 10
        self.num_actions = 0
        self.ckpt_time = time.time_ns()

    def expand_node(self, node: Node, network_output):  # takes negligible time
        node.to_play = 0
        node.hidden_state = network_output["hidden_state"]
        node.reward = network_output["reward"]

        # policy_probs = np.array(masked_softmax(network_output["policy_logits"].numpy()[0]))
        policy_probs = network_output["policy_logits"].numpy()[0]

        policy = expand_node2(policy_probs, config.ACTION_DIM)
        # This policy sum is not in the Google's implementation. Not sure if required.
        policy_sum = sum(policy[0].values())
        for action, p in policy[0].items():
            node.children[action] = Node(p / policy_sum)

    def expand_node_old(self, node: Node, network_output):  # old version of expand_node for a failsafe
        policy = {b: math.exp(network_output["policy_logits"][0][b]) for b in range(self.action_dim)}
        policy_sum = sum(policy.values())
        for action, p in policy.items():
            node.children[action] = Node(p / policy_sum)

    def add_exploration_noise(self, node: dict, key):  # takes 0 time
        actions = list(node[key].children.keys())
        noise = np.random.dirichlet([config.ROOT_DIRICHLET_ALPHA] * len(actions))
        frac = config.ROOT_EXPLORATION_FRACTION
        for a, n in zip(actions, noise):
            node[node[key].children[a]].prior = node[node[key].children[a]].prior * (1 - frac) + n * frac

    # Select the child with the highest UCB score.
    def select_child(self, node: dict, key, min_max_stats: MinMaxStats):
        _, action, max_child = max((self.ucb_score(node, child, min_max_stats, key), action,
                                child) for action, child in node[key].children.items())
        return action, max_child

    # The score for a node is based on its value, plus an exploration bonus based on
    # the prior.
    @staticmethod
    def ucb_score(node: dict, child: int, min_max_stats: MinMaxStats, key) -> float:  # Takes aprx 0 time
        pb_c = math.log((node[key].visit_count + config.PB_C_BASE + 1) /
                        config.PB_C_BASE) + config.PB_C_INIT
        pb_c *= math.sqrt(node[key].visit_count) / (node[child].visit_count + 1)
        prior_score = pb_c * node[child].prior
        value_score = min_max_stats.normalize(node[child].value())
        return prior_score + value_score

    # At the end of a simulation, we propagate the evaluation all the way up the
    # tree to the root.
    @staticmethod
    def backpropagate(node: dict, search_path: List[dict], value: float,
                      min_max_stats: MinMaxStats):  # takes lots of time
        last_value = value
        for key in search_path:
            
            node[key].value_sum += last_value  # 2.72s
            node[key].visit_count += 1
        
            min_max_stats.update(node[key].value())  # 1.48s
            
            last_value = node[key].reward + config.DISCOUNT * value  # 1.76s

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

            network_output = self.network. \
                recurrent_inference(parent.hidden_state, np.expand_dims(np.asarray(history.last_action()), axis=0))
            self.expand_node(node, network_output)
            self.backpropagate(search_path, network_output["value"], min_max_stats)

    def select_action(self, node: dict):
        visit_counts = [
            (node[child].visit_count, action) for action, child in node[0].children.items()
        ]
        t = self.visit_softmax_temperature()
        return self.histogram_sample(visit_counts, t, use_softmax=False)

    def policy(self, observation, previous_action):
        root = Node(0)

        network_output = self.network.initial_inference(observation)
        self.expand_node(root, network_output)
        self.add_exploration_noise(root)

        self.run_mcts(root, network_output["policy_logits"].numpy(), previous_action)

        action = int(self.select_action(root))

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
        return .9


class Batch_MCTSAgent(MCTSAgent):
    """
    Use Monte-Carlo Tree-Search to select moves.
    """

    def __init__(self, network: Network) -> None:
        super().__init__(network, 0)
        self.NUM_ALIVE = config.NUM_PLAYERS

    # Core Monte Carlo Tree Search algorithm.
    # To decide on an action, we run N simulations, always starting at the root of
    # the search tree and traversing the tree according to the UCB formula until we
    # reach a leaf node.
    def run_batch_mcts(self, node: list, action: List):
        # run_batch_mcts took 5.652412176132202 seconds to finish
        min_max_stats = [MinMaxStats(config.MINIMUM_REWARD, config.MAXIMUM_REWARD) for _ in range(config.NUM_PLAYERS)]
        
        for _ in range(config.NUM_SIMULATIONS):
            history = [ActionHistory(action[i]) for i in range(self.NUM_ALIVE)]
            search_path = [[0] for i in range(self.NUM_ALIVE)]
            
            # There is a chance I am supposed to check if the tree for the non-main-branch
            # Decision paths (axis 1-4) should be expanded. I am currently only expanding on the
            # main decision axis.
            
            for i in range(self.NUM_ALIVE):
                curr_node = 0 #Starting at root
                while node[i][curr_node].expanded():
                    selected_action, curr_node = self.select_child(node[i], curr_node, min_max_stats[i])
                    history[i].add_action(selected_action)
                    search_path[i].append(curr_node)
            
            # Inside the search tree we use the dynamics function to obtain the next
            # hidden state given an action and the previous hidden state.
            parent = [search_path[i][-2] for i in range(self.NUM_ALIVE)]
            hidden_state = np.asarray([node[i][parent[i]].hidden_state for i in range(self.NUM_ALIVE)])
            last_action = [history[i].last_action() for i in range(self.NUM_ALIVE)]

            network_output = self.network.recurrent_inference(hidden_state, last_action)  # 11.05s
            for i in range(self.NUM_ALIVE):
                
                self.batch_expand_node(node[i], node[i][search_path[i][-1]].to_play,  network_output, search_path[i][-1])  # 7s
                
                # print("value {}".format(network_output["value"]))
                self.backpropagate(node[i], search_path[i], network_output["value"].numpy()[i], min_max_stats[i])  # 12.08s

    def batch_policy(self, observation, prev_action):
        # batch_policy took 6.422544717788696 seconds to finish
        # observation.shape = (8, 8246)
        self.NUM_ALIVE = observation.shape[0]
        root = [{0: Node(0)} for _ in range(self.NUM_ALIVE)]
        network_output = self.network.initial_inference(observation)  # 2.1 seconds
        
        for i in range(self.NUM_ALIVE):
            self.batch_expand_node(root[i], i, network_output, 0, obs=observation[i])  # 0.39 seconds
            self.add_exploration_noise(root[i], 0)
            
        self.run_batch_mcts(root, prev_action)  # 24.3 s (3 seconds not in that)

        action = [(self.select_action(root[i])) for i in range(self.NUM_ALIVE)]
        
        # Masking only if training is based on the actions taken in the environment.
        # Training in MuZero is mostly based on the predicted actions rather than the real ones.
        # network_output["policy_logits"], action = self.maskInput(network_output["policy_logits"], action)

        # Notes on possibilities for other dimensions at the bottom
        self.num_actions += 1
       
        return action, network_output["policy_logits"]

    def policy(self, observation, prev_action):
        # observation.shape = (8246)
        root = {0: Node(0)}
        network_output = self.network.initial_inference(observation.reshape(1,8246))
        
        self.expand_node(root, network_output, 0, obs=observation)
        self.add_exploration_noise(root, 0)
            
        self.run_mcts(root, prev_action)  # 24.3 s (3 seconds not in that)

        action = self.select_action(root)
        
        # Masking only if training is based on the actions taken in the environment.
        # Training in MuZero is mostly based on the predicted actions rather than the real ones.
        # network_output["policy_logits"], action = self.maskInput(network_output["policy_logits"], action)

        # Notes on possibilities for other dimensions at the bottom
        self.num_actions += 1
       
        return action, network_output["policy_logits"]

    def batch_expand_node(self, node: dict, to_play: int, network_output, key, obs=None):
        # batch_expand_node took 0.021803855895996094 seconds to finish
        node[key].to_play = to_play
        node[key].hidden_state = network_output["hidden_state"][to_play]
        node[key].reward = network_output["reward"][to_play]

        policy_action_probs = network_output["policy_logits"][0][to_play].numpy()
        policy_target_probs = network_output["policy_logits"][1][to_play].numpy()
        policy_item_probs = network_output["policy_logits"][2][to_play].numpy()

        policy_action = expand_node2(policy_action_probs, config.ACTION_DIM[0])
        policy_target = expand_node2(policy_target_probs, config.ACTION_DIM[1])
        policy_item = expand_node2(policy_item_probs, config.ACTION_DIM[2])
        # This policy sum is not in the Google's implementation. Not sure if required.
        policy_sum = sum(policy_action[0].values()) + sum(policy_target[0].values()) + sum(policy_item[0].values())

        actions_p = self.encode_action_to_str(policy_action, policy_target, policy_item, obs)
        for action, p in actions_p:
            children_key = len(node.keys())
            node[key].children[action] = children_key
            node[children_key] = Node(p / policy_sum)
    
    def expand_node(self, node: dict, network_output, key, obs=None):
        # batch_expand_node took 0.021803855895996094 seconds to finish
        node[key].to_play = 0
        node[key].hidden_state = network_output["hidden_state"]
        node[key].reward = network_output["reward"]

        policy_action_probs = network_output["policy_logits"][0].numpy()
        policy_target_probs = network_output["policy_logits"][1].numpy()
        policy_item_probs = network_output["policy_logits"][2].numpy()

        policy_action = expand_node2(policy_action_probs.reshape(config.ACTION_DIM[0],), config.ACTION_DIM[0])
        policy_target = expand_node2(policy_target_probs.reshape(config.ACTION_DIM[1],), config.ACTION_DIM[1])
        policy_item = expand_node2(policy_item_probs.reshape(config.ACTION_DIM[2],), config.ACTION_DIM[2])
        # This policy sum is not in the Google's implementation. Not sure if required.
        policy_sum = sum(policy_action[0].values()) + sum(policy_target[0].values()) + sum(policy_item[0].values())

        actions_p = self.encode_action_to_str(policy_action, policy_target, policy_item, obs)
        for action, p in actions_p:
            children_key = len(node.keys())
            node[key].children[action] = children_key
            node[children_key] = Node(p / policy_sum)

    def run_mcts(self, node: list, action):
        # run_batch_mcts took 5.652412176132202 seconds to finish
        min_max_stats = MinMaxStats(config.MINIMUM_REWARD, config.MAXIMUM_REWARD)
        
        for _ in range(config.NUM_SIMULATIONS):
            history = ActionHistory(action)
            search_path = [0]
            
            # There is a chance I am supposed to check if the tree for the non-main-branch
            # Decision paths (axis 1-4) should be expanded. I am currently only expanding on the
            # main decision axis.
            
            curr_node = 0 #Starting at root
            while node[curr_node].expanded():
                selected_action, curr_node = self.select_child(node, curr_node, min_max_stats)
                history.add_action(selected_action)
                search_path.append(curr_node)
            
            # Inside the search tree we use the dynamics function to obtain the next
            # hidden state given an action and the previous hidden state.
            parent = search_path[-2]
            hidden_state = np.asarray(node[parent].hidden_state)
            last_action = history.last_action()

            network_output = self.network.recurrent_inference(hidden_state, [last_action])  # 11.05s
            self.expand_node(node, network_output, search_path[-1])
            
            self.backpropagate(node, search_path, network_output["value"].numpy(), min_max_stats)

    def encode_action_to_str(self, action, target, item, obs=None):
        gold = 100
        level = 10
        unit_count = 0
        shop = [True for _ in range(5)]
        shop_price = [0 for _ in range(5)]
        board_bench = [True for _ in range(37)]
        items = [True for _ in range(10)]
        if obs is not None:
            gold = int(obs[1027])
            level = int(obs[1])
            shop = [np.any(obs[(1029 + i*7):(1036 + i*7)]) for i in range(5)]
            list_champs = list(COST.keys())
            shop_price = [COST[list_champs[utils.champ_binary_decode(obs[(1029 + i*7):(1036 + i*7)])]] for i in range(5)]
            board_bench = [np.any(obs[(4 + i*26):(30 + i*26)]) for i in range(37)]
            for unit in board_bench[0:28]:
                if unit:
                    unit_count += 1
            items = [np.any(obs[(966 + i*6):(972 + i*6)]) for i in range(10)]
        board_bench.append(False)

        actions = []
        actions.append(("0",action[0][0]))
        for i in range(5): #TODO check if there is space in bench
            if shop[i] and gold >= shop_price[i]:
                actions.append((f"1_{i}",action[0][1] * target[0][i] / sum(list(target[0].values())[0:5])))
        for a in range(37):
            for b in range(a, 38):
                if a == b:
                    continue
                if board_bench[a] or (board_bench[b] and unit_count < level):
                    actions.append((f"2_{a}_{b}",action[0][2] * 
                                    (target[0][a] / sum(list(target[0].values())[0:37]) *
                                    (target[0][b] / (sum(list(target[0].values())[0:38])  - target[0][a])))))
        for a in range(37):
            for b in range(10):
                if board_bench[a] and items[b]:
                    actions.append((f"3_{a}_{b}",action[0][3] *
                                    (target[0][a] / sum(list(target[0].values())[0:37])) * (item[0][b] / sum(list(item[0].values())))))
        if gold >= 4:
            actions.append(("4",action[0][4]))
        if gold >= 2:
            actions.append(("5",action[0][5]))
        return actions

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
