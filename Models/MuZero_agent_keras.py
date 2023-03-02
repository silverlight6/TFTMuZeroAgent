import collections
import numpy as np
import tensorflow as tf
import time
import Simulator.utils as utils
from Simulator.stats import COST
from Simulator.pool_stats import cost_star_values
import config

##########################
####### Helpers ##########

MAXIMUM_FLOAT_VALUE = float('inf')

KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])

NetworkOutput = collections.namedtuple(
    'NetworkOutput',
    'value reward policy_logits hidden_state')

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
        reward_output = tf.keras.layers.Dense(units=601, name='reward', activation='relu', kernel_regularizer=regularizer,
                                              bias_regularizer=regularizer)(rnn_output)

        dynamics_model: tf.keras.Model = \
            tf.keras.Model(inputs=[dynamic_hidden_state, encoded_state_action],
                           outputs=[next_hidden_state, reward_output], name='dynamics')

        pred_hidden_state = tf.keras.Input(shape=np.array([2 * config.HIDDEN_STATE_SIZE]), name="prediction_input")
        x = Mlp(hidden_size=config.HIDDEN_STATE_SIZE, mlp_dim=config.HEAD_HIDDEN_SIZE)(pred_hidden_state)
        value_x = Mlp(hidden_size=config.HIDDEN_STATE_SIZE, mlp_dim=config.HEAD_HIDDEN_SIZE)(x)
        value_output = tf.keras.layers.Dense(units=601, name='value', activation='relu', kernel_regularizer=regularizer,
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