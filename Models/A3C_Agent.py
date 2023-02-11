import tensorflow as tf
import numpy as np
from datetime import datetime

import config

class A3C_Agent:
    def __init__(self, t_board=None):
        self.optimizer = tf.optimizers.RMSprop(learning_rate=1e-3)
        self.action_dim = config.ACTION_8D_DIM
        self.mlp_block_num = 2
        self.popArtLayer = PopartLayer(1, hidden_dim=sum(self.action_dim) + config.HEAD_HIDDEN_SIZE)
        self.batch_size = config.BATCH_SIZE
        self.sequence_length = config.A3C_SEQUENCE_LENGTH
        self.a3c_net = self.build_a3c_model()
        self.gamma = 0.99
        self.t_board = t_board
        # print(self.a3c_net.summary())
        logs = "./logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.file_writer = tf.summary.create_file_writer(logs)
        self.file_writer.set_as_default()

    def build_a3c_model(self):
        # Representation model. Observation --> hidden state
        rep_tensor_input = tf.keras.Input(shape=config.INPUT_TENSOR_SHAPE)
        tensor_x = Mlp(hidden_size=config.HIDDEN_STATE_SIZE / 2, name="rep_tensor")(rep_tensor_input)
        tensor_x = tf.keras.layers.Dense(config.HIDDEN_STATE_SIZE, activation='sigmoid', name='rep_tensor')(tensor_x)

        rep_image_input = tf.keras.Input(shape=config.INPUT_IMAGE_SHAPE)
        image_x = tf.keras.layers.Conv2D(filters=config.CONV_FILTERS, kernel_size=4, strides=(2, 2),
                                         padding='same', use_bias=False, name='conv_resize')(rep_image_input)
        image_x = ResidualBlock(config.CONV_FILTERS)(image_x)
        image_x = ResidualBlock(config.CONV_FILTERS)(image_x)
        image_x = tf.keras.layers.Flatten()(image_x)
        image_x = tf.keras.layers.Dense(config.HIDDEN_STATE_SIZE, activation='sigmoid', name='rep_image')(image_x)

        rep_output = tf.concat([tensor_x, image_x], axis=-1)

        x = tf.expand_dims(rep_output, axis=1)

        x = tf.keras.layers.LSTM(1024, activation='tanh', name='lstm_layer')(x)

        # Adding in the previous action and reward to the end of the LSTM output
        prev_stats = tf.keras.Input(shape=[sum(self.action_dim)], dtype=np.float32)
        x = tf.concat([x, prev_stats], axis=-1)

        # actor_layer = tf.keras.layers.Dense(256, activation='relu')(x)
        decision_output = tf.keras.layers.Dense(self.action_dim[0], activation='softmax', name='decision_layer')(x)
        shop_output = tf.keras.layers.Dense(self.action_dim[1], activation='softmax', name='shop_layer')(x)
        bench_output = tf.keras.layers.Dense(self.action_dim[2], activation='softmax', name='bench_layer')(x)
        item_output = tf.keras.layers.Dense(self.action_dim[3], activation='softmax', name='item_layer')(x)
        board_output_x = tf.keras.layers.Dense(self.action_dim[4], activation='softmax', name='board_x_layer')(x)
        board_output_y = tf.keras.layers.Dense(self.action_dim[5], activation='softmax', name='board_y_layer')(x)
        board_output_x2 = tf.keras.layers.Dense(self.action_dim[6], activation='softmax', name='board_x2_layer')(x)
        board_output_y2 = tf.keras.layers.Dense(self.action_dim[7], activation='softmax', name='board_y2_layer')(x)
        # critic_layer = tf.keras.layers.Dense(256, activation='relu')(x)
        norm_v, un_norm_v = self.popArtLayer(x)
        output = tf.keras.Model(inputs=[rep_tensor_input, rep_image_input, prev_stats],
                                outputs=[[decision_output, shop_output, item_output, bench_output, board_output_x,
                                          board_output_y, board_output_x2, board_output_y2],
                                         [norm_v, un_norm_v]])
        return output

    def batch_policy(self, observation, prev_action):
        one_hot_last_action = self.action_one_hot(prev_action)

        action_q, [value, _] = self.a3c_net([observation[0], observation[1], one_hot_last_action])

        actions = np.swapaxes(np.asarray([tf.random.categorical(action_q[:][i], 1).numpy()
                              for i in range(len(self.action_dim))]), 0, 1)

        actions = np.squeeze(actions)

        return actions, value

    def action_one_hot(self, batch_actions):
        o_h_last_action = []
        for b in range(len(batch_actions)):
            one_hot_last_action = tf.Variable(tf.one_hot(batch_actions[b][0], self.action_dim[0]))
            for i in range(1, len(self.action_dim)):
                one_hot_last_action = tf.concat([one_hot_last_action,
                                                 tf.one_hot(batch_actions[b][i], self.action_dim[i])], axis=0)
            o_h_last_action.append(one_hot_last_action)
        return tf.convert_to_tensor(o_h_last_action)

    # Renaming as to not override built-in functions
    def tft_save_model(self, episode):
        self.a3c_net.save_weights("./Checkpoints/checkpoint_{}".format(episode))

    # Renaming as to not override built-in functions
    def tft_load_model(self, episode):
        self.a3c_net.load_weights("./Checkpoints/checkpoint_{}".format(episode))
        print("Loading model episode {}".format(episode))


class PopartLayer(tf.keras.layers.Layer):
    def __init__(self, num_actions, hidden_dim=config.HEAD_HIDDEN_SIZE):
        super(PopartLayer, self).__init__()
        self._num_actions = num_actions
        self.mean = tf.Variable([0], dtype=tf.float32, trainable=False)
        self.mean_squared = tf.Variable([0], dtype=tf.float32, trainable=False)
        self.std = tf.Variable([1], dtype=tf.float32, trainable=False)
        self._beta = 3e-4
        self._epsilon = 1e-4
        self.hidden_dim = hidden_dim
        layer_init = tf.random_normal_initializer()
        self.weight = self.add_weight("pop_kernel", shape=(self.hidden_dim, self._num_actions),
                                      trainable=True, initializer=layer_init)
        self.bias = self.add_weight("pop_bias", shape=(self._num_actions,),
                                    trainable=True, initializer=layer_init)

    def call(self, inputs):
        head_output = tf.matmul(inputs, self.weight) + self.bias
        un_norm_output = self.std * head_output + self.mean
        return head_output, un_norm_output

    def forward(self, x):
        head_output = tf.matmul(x, self.weight) + self.bias
        un_norm_output = self.std * head_output + self.mean
        return head_output, un_norm_output

    def update_moments(self, vs):

        def update_step(mm, _tuple):
            mean, mean_squared = mm
            vss = _tuple

            # According to equation (6) in the PopArt-IMPALA paper
            # Matching the specific game with it's current vtrace corrected value estimate.
            first_moment = tf.reshape((1 - self._beta) * mean + self._beta * vss, [1])
            second_moment = tf.reshape((1 - self._beta) * mean_squared + self._beta * tf.square(vss), [1])

            return first_moment, second_moment

        def update_batch(mm, gvt):
            return tf.foldl(update_step, gvt, initializer=mm)

        new_mean, new_mean_squared = tf.foldl(update_batch, vs, initializer=(self.mean, self.mean_squared))
        new_std = tf.sqrt(new_mean_squared - tf.square(new_mean))
        new_std = tf.clip_by_value(new_std, self._epsilon, 1e6)

        # According to equation (9) in (Hessel et al., 2018)

        self.weight = self.weight * self.std / new_std
        self.bias = (self.std * self.bias + self.mean - new_mean) / new_std

        self.mean = new_mean
        self.mean_squared = new_mean_squared
        self.std = new_std

        return self.mean, self.mean_squared

    
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
