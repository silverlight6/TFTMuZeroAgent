from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import List
import collections
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
        tensor_observation = tf.keras.Input(shape=config.INPUT_TENSOR_SHAPE, dtype=tf.float32, name='t_observation')

        hidden_state = self.representation(tensor_observation)

        value, policy_logits = self.prediction(hidden_state)

        return tf.keras.Model(inputs=tensor_observation,
                              outputs=[hidden_state, value, policy_logits],
                              name='initial_inference')

    # Apply the initial inference model to the given hidden state
    def initial_inference(self, observation) -> dict:
        hidden_state, value_logits, policy_logits = \
            self.initial_inference_model(observation, training=True)
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
        hidden_state = tf.keras.Input(shape=[config.LAYER_HIDDEN_SIZE], dtype=tf.float32, name='hidden_state')
        # state_space = tf.keras.Input(shape=[1, config.HIDDEN_STATE_SIZE], dtype=tf.float32, name='state_space')
        action = tf.keras.Input(shape=([config.ACTION_CONCAT_SIZE]), dtype=tf.int32, name='action')

        new_hidden_state, reward = self.dynamics((hidden_state, action))

        value, policy_logits = self.prediction(new_hidden_state)

        return tf.keras.Model(inputs=[hidden_state, action],
                              outputs=[new_hidden_state, reward, value, policy_logits],
                              name='recurrent_inference')

    # Apply the recurrent inference model to the given hidden state
    def recurrent_inference(self, hidden_state, action) -> dict:
        one_hot_action = tf.one_hot(action[:, 0], config.ACTION_DIM[0], 1., 0., axis=-1)
        one_hot_target_a = tf.one_hot(action[:, 1], config.ACTION_DIM[1], 1., 0., axis=-1)
        one_hot_target_b = tf.one_hot(action[:, 2], config.ACTION_DIM[1] - 1, 1., 0., axis=-1)

        final_action = tf.concat([one_hot_action, one_hot_target_a, one_hot_target_b], axis=-1)
        hidden_state, reward_logits, value_logits, policy_logits = \
            self.recurrent_inference_model((hidden_state, final_action), training=True)

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
        for size in config.RNN_SIZES:
            states = (state[Ellipsis, cur_idx:cur_idx + size],
                      state[Ellipsis, cur_idx + size:cur_idx + 2 * size])
            cur_idx += 2 * size
            tensors.append(states)
        assert cur_idx == state.shape[-1]
        return tensors


class TFTNetwork(Network):
    """
    Neural networks for tic-tac-toe game.
    """

    def __init__(self) -> None:
        regularizer = tf.keras.regularizers.l2(l=1e-4)

        # Representation model. Convert observation to hidden state
        rep_tensor_input = tf.keras.Input(shape=config.INPUT_TENSOR_SHAPE)
        tensor_x = Mlp(hidden_size=config.HIDDEN_STATE_SIZE / 2, name="rep_tensor")(rep_tensor_input)
        rep_output = tf.keras.layers.Dense(config.LAYER_HIDDEN_SIZE, activation='sigmoid', name='rep_tensor')(tensor_x)

        # Using representation model from tf.keras.Model
        representation_model: tf.keras.Model = \
            tf.keras.Model(inputs=rep_tensor_input,
                           outputs=rep_output, name='observation_encodings')

        # Dynamics Model. Hidden State --> next hidden state and reward
        # Action encoding
        encoded_state_action = tf.keras.Input(shape=[config.ACTION_CONCAT_SIZE])
        action_embeddings = tf.keras.layers.Dense(units=config.LAYER_HIDDEN_SIZE, activation='relu',
                                                  kernel_regularizer=regularizer,
                                                  bias_regularizer=regularizer)(encoded_state_action)
        action_embeddings = tf.keras.layers.Flatten()(action_embeddings)

        # Hidden state input. [[1, 256], [1. 256]] Needs both the hidden state and lstm state
        dynamic_hidden_state = tf.keras.Input(shape=[config.LAYER_HIDDEN_SIZE], name='hidden_state_input')
        rnn_state = self.flat_to_lstm_input(dynamic_hidden_state)

        # Core of the model
        rnn_cell_cls = {
            'lstm': tf.keras.layers.LSTMCell,
        }['lstm']
        rnn_cells = [
            rnn_cell_cls(
                size,
                recurrent_activation='sigmoid',
                name='cell_{}'.format(idx)) for idx, size in enumerate(config.RNN_SIZES)]
        core = tf.keras.layers.StackedRNNCells(rnn_cells, name='recurrent_core')

        rnn_output, next_rnn_state = core(action_embeddings, rnn_state)
        next_hidden_state = self.rnn_to_flat(next_rnn_state)

        # Reward head
        reward_output = tf.keras.layers.Dense(units=config.ENCODER_NUM_STEPS, name='reward',
                                              kernel_regularizer=regularizer, bias_regularizer=regularizer)(rnn_output)
        dynamics_model: tf.keras.Model = \
            tf.keras.Model(inputs=[dynamic_hidden_state, encoded_state_action],
                           outputs=[next_hidden_state, reward_output], name='dynamics')

        pred_hidden_state = tf.keras.Input(shape=np.array([config.LAYER_HIDDEN_SIZE]), name="prediction_input")
        value_x = Mlp(hidden_size=config.HIDDEN_STATE_SIZE, name="value")(pred_hidden_state)
        value_output = tf.keras.layers.Dense(units=config.ENCODER_NUM_STEPS, name='value',
                                             kernel_regularizer=regularizer, bias_regularizer=regularizer)(value_x)

        policy_x = Mlp(hidden_size=config.HIDDEN_STATE_SIZE, name="policy")(pred_hidden_state)
        policy_output_action = tf.keras.layers.Dense(config.ACTION_ENCODING_SIZE,
                                                     name='policy_output')(policy_x)

        prediction_model: tf.keras.Model = tf.keras.Model(inputs=pred_hidden_state,
                                                          outputs=[value_output, policy_output_action],
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
    def __init__(self, num_layers=2, hidden_size=512, name=""):
        super(Mlp, self).__init__()

        # Default input gives two layers: [1024, 512]
        sizes = [hidden_size * layer for layer in range(num_layers, 0, -1)]
        layers = []

        for size in sizes:
            layers.extend([
                tf.keras.layers.Dense(size, dtype=tf.float32),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.ReLU()
            ])

        self.net = tf.keras.Sequential(layers, name=name + "_mlp")

    def forward(self, x):
        return self.net(x)

    def __call__(self, x, *args, **kwargs):
        out = self.forward(x)
        return out


class ResidualBlock(tf.keras.layers.Layer):
    """
    Residual block.
    Implementation adapted from:
    https://towardsdatascience.com/from-scratch-implementation-of-alphazero-for-connect4-f73d4554002a
    .
    """

    def __init__(self, planes):
        super(ResidualBlock, self).__init__(name='')
        self.planes = planes

        # Question mark if we want to use kernel size 3 or 4 given our board slots are 7x4 tensors.
        self.conv2a = tf.keras.layers.Conv2D(
            filters=self.planes,
            kernel_size=4,
            strides=(1, 1),
            padding='same',
            use_bias=False)
        self.bn2a = tf.keras.layers.LayerNormalization()

        self.conv2b = tf.keras.layers.Conv2D(
            filters=self.planes,
            kernel_size=4,
            strides=(1, 1),
            padding='same',
            use_bias=False)
        self.bn2b = tf.keras.layers.LayerNormalization()
        self.relu = tf.keras.layers.ReLU()

    def __call__(self, input_tensor, training=True, **kwargs):
        x = self.conv2a(input_tensor, training=training)
        x = self.bn2a(x, training=training)
        x = self.relu(x)

        x = self.conv2b(x, training=training)
        x = self.bn2b(x, training=training)

        x += input_tensor
        return self.relu(x)


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


