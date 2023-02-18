from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Dict
import collections
import config
import time
import tensorflow as tf

NetworkOutput = collections.namedtuple(
    'NetworkOutput',
    'value reward policy_logits hidden_state')


class MuZero_agent(tf.keras.Model):

    def __init__(self):
        super().__init__(name='MuZeroAgent')
        self.ckpt_time = time.time_ns()

        self.action_dim = config.ACTION_DIM
        self.batch_size = config.BATCH_SIZE
        self.head_hidden_sizes = [config.HEAD_HIDDEN_SIZE] * config.N_HEAD_HIDDEN_LAYERS
        self.res_conv_sizes = [config.CONV_FILTERS] * config.N_HEAD_HIDDEN_LAYERS
        self.num_actions = 0
        self.mlp1 = Mlp(hidden_size=config.HIDDEN_STATE_SIZE, mlp_dim=config.HEAD_HIDDEN_SIZE)
        self.mlp2 = Mlp(hidden_size=config.HIDDEN_STATE_SIZE, mlp_dim=config.HEAD_HIDDEN_SIZE)
        self.init_conv = tf.keras.layers.Conv2D(filters=config.CONV_FILTERS, kernel_size=4, strides=(2, 2),
                                                padding='same', use_bias=False, name='conv_resize')
        self.res1 = ResidualBlock(config.CONV_FILTERS)

        rnn_cell_cls = {
            'lstm': tf.keras.layers.LSTMCell,
        }['lstm']
        rnn_cells = [
            rnn_cell_cls(
                size,
                recurrent_activation='sigmoid',
                name='cell_{}'.format(idx)) for idx, size in enumerate(config.RNN_SIZES)]
        self._core = tf.keras.layers.StackedRNNCells(rnn_cells, name='recurrent_core')

        # TO DO: Move to utilities
        self.value_encoder = ValueEncoder(*tuple(map(inverse_contractive_mapping, (-300., 300.))))

        self.reward_encoder = ValueEncoder(*tuple(map(inverse_contractive_mapping, (-300., 300.))))

        self._to_hidden = tf.keras.layers.Dense(config.HIDDEN_STATE_SIZE, activation='sigmoid', name='final')
        self._value_head = tf.keras.layers.Dense(config.ENCODER_NUM_STEPS, name='output', dtype=tf.float32)
        self._reward_head = tf.keras.layers.Dense(config.ENCODER_NUM_STEPS, name='output', dtype=tf.float32)

        self.policy_output_action = tf.keras.layers.Dense(config.ACTION_DIM[0], name='action_layer')
        self.policy_output_target = tf.keras.layers.Dense(config.ACTION_DIM[1], name='target_layer')
        self.policy_output_item = tf.keras.layers.Dense(config.ACTION_DIM[2], name='item_layer')

    def initial_inference(self, observation) -> Dict:
        # representation + prediction function
        encoded_observation = self.encode_observation(
            observation)
        hidden_state = self.to_hidden(encoded_observation)

        # could add encoding but more research has to be done to why that is a good idea
        value_logits = self.value_head(hidden_state)
        value_softmax = tf.nn.softmax(value_logits)
        value = self.value_encoder.decode(value_softmax)

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
        one_hot_action = tf.one_hot(action[:, 0], config.ACTION_DIM[0], 1., 0., axis=-1)
        one_hot_target_a = tf.one_hot(action[:, 1], config.ACTION_DIM[1], 1., 0., axis=-1)
        one_hot_target_b = tf.one_hot(action[:, 2], config.ACTION_DIM[1] - 1, 1., 0., axis=-1)

        final_action = tf.concat([one_hot_action, one_hot_target_a, one_hot_target_b], axis=-1)

        embedded_action = self.action_embeddings(final_action)
        rnn_state = self.flat_to_lstm_input(hidden_state)

        rnn_output, next_rnn_state = self.core(embedded_action, rnn_state)

        next_hidden_state = self.rnn_to_flat(next_rnn_state)

        # could add encoding but more research has to be done to why that is a good idea
        value_logits = self.value_head(next_hidden_state)
        value_softmax = tf.nn.softmax(value_logits)
        value = self.value_encoder.decode(value_softmax)

        # Rewards are only calculated in recurrent_inference.
        reward_logits = self.reward_head(rnn_output)
        reward_softmax = tf.nn.softmax(reward_logits)
        reward = self.reward_encoder.decode(reward_softmax)

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
        # not sure if 1 or 2 mlp blocks are needed
        tensor = self.mlp1(observation[0])
        tensor = self.mlp2(tensor)

        image = self.init_conv(observation[1])
        image = self.conv_layers(image)
        flatten_image = tf.keras.layers.Flatten()(image)

        x = tf.concat([tensor, flatten_image], axis=-1)
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
        action_output = self.policy_output_action(x)
        target_output = self.policy_output_target(x)
        item_output = self.policy_output_item(x)
        return [action_output, target_output, item_output]

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

    def conv_layers(self, x):
        def _make_layer():
            return ResidualBlock(config.CONV_FILTERS)

        for idx, size in enumerate(self.res_conv_sizes):
            x = tf.keras.Sequential(_make_layer(), name='res_block_{}'.format(idx))(x)

        return x

    def flat_to_lstm_input(self, state):
        """Maps flat vector to LSTM state."""
        tensors = []
        cur_idx = 0
        for size in config.RNN_SIZES:
            states = (state[Ellipsis, cur_idx:cur_idx + size],
                      state[Ellipsis, cur_idx + size:cur_idx + 2 * size])
            cur_idx += 2 * size
            tensors.append(states)
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


class ValueEncoder:
    """Encoder for reward and value targets from Appendix of MuZero Paper."""

    def __init__(self,
                 min_value,
                 max_value,
                 use_contractive_mapping=True):
        if use_contractive_mapping:
            max_value = contractive_mapping(max_value)
            min_value = contractive_mapping(min_value)
        num_steps = tf.math.ceil(max_value) + 1 - tf.math.floor(min_value)
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
