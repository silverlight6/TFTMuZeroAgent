import tensorflow as tf
import numpy as np


class DQN_Agent:
    def __init__(self, input_shape):
        self.q_net = self.build_dqn_model(input_shape)
        self.target_net = self.build_dqn_model(input_shape)

    def collect_policy(self, state, episode):
        if np.random.random() < 0.2 - (episode / 5000):
            return self.random_policy(state)
        else:
            return self.policy(state)

    def policy(self, state):
        # state is the observation in this case.
        # this is the policy network.
        state_input = tf.convert_to_tensor(state[None, :], dtype=tf.float32)

        action_q = self.q_net(state_input)
        # probably going to have some issues with this line since my return is 2 dimensional.
        action = []
        for i in range(len(action_q)):
            action.append(np.argmax(action_q[i].numpy()))
        # action = tf.math.argmax(action_q, axis=2).numpy()
        # action = np.argmax(action_q.numpy()[0], axis=0)

        return action

    # This can be updated as well so that the random policy is applied to some components of my action
    # and not other actions.
    def random_policy(self, state):
        return [np.random.randint(0, 12), np.random.randint(0, 10), np.random.randint(0, 9),
                np.random.randint(0, 7), np.random.randint(0, 4)]

    def update_target_network(self):
        self.target_net.set_weights(self.q_net.get_weights())

    def train(self, batch):
        # Trains the underlying network with batches of gameplay experience
        # Returns a training loss

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch
        # print(state_batch.shape)
        error = False
        for i in range(state_batch.shape[0] - 1, -1, -1):
            if len(state_batch[i][0]) != 1109:
                # Putting this here for now and will solve the optimality issues later.
                # print("ERROR")
                print("When i = " + str(i) + " error occurs with length = " + str(len(state_batch[i][0])))
                print(type(state_batch[i][0]))
                error = True
                state_batch = np.delete(state_batch, i, 0)
                action_batch = np.delete(action_batch, i, 0)
                reward_batch = np.delete(reward_batch, i, 0)
                next_state_batch = np.delete(next_state_batch, i, 0)
                done_batch = np.delete(done_batch, i, 0)
                # print("Done at i = " + str(done_batch[i]))
                # print("with the total length of the array after the error being " + str(state_batch.shape))
                # print(len(state_batch[i-1][0]))

        # print(len(state_batch[0][0]))
        if error:
            tmp_state_batch = []
            tmp_next_state_batch = []
            for i in range(0, state_batch.shape[0]):
                tmp_state_batch.append(state_batch[i][0])
                tmp_next_state_batch.append(next_state_batch[i][0])
                # done_batch[i] = False
            # done_batch[state_batch.shape[0] - 1] = True
            state_batch = np.array(tmp_state_batch)
            state_batch = np.reshape(state_batch, (state_batch.shape[0], 1, 1109))
            next_state_batch = np.array(tmp_next_state_batch)
            next_state_batch = np.reshape(next_state_batch, (next_state_batch.shape[0], 1, 1109))
            # print(done_batch)

        # state_batch = np.reshape(state_batch, newshape=(state_batch.shape[0], 1, 1094))
        # print(state_batch.shape)
        # state_batch = tf.convert_to_tensor(state_batch, dtype=tf.float32)
        current_q = self.q_net(state_batch)
        current_q = list(current_q)
        target_q = []
        for i in range(0, 5):
            target_q.append(tf.identity(current_q[i]).numpy())
        next_q = self.target_net(next_state_batch)
        max_next_q = []
        # print("len of next_q = " + str(len(next_q[0])))
        for b in range(0, len(next_q[0])):
            max_next_q.append([np.amax(next_q[0].numpy()), np.amax(next_q[1].numpy()), np.amax(next_q[2].numpy()),
                               np.amax(next_q[3].numpy()), np.amax(next_q[4].numpy())])
        # print("Length of my tensors' second dimension")
        # print(action_batch.shape)
        # print(len(target_q[1]))
        # print(reward_batch.shape)
        # print(len(max_next_q))
        # print(len(max_next_q[0]))
        # print(target_q[0].shape)
        for idx in range(0, len(target_q[1])):
            for field in range(0, 5):
                if not done_batch[idx]:
                    target_q[field][idx][0][action_batch[idx][field]] = reward_batch[idx] + 0.95 * max_next_q[idx][field]
                else:
                    # print("In the done = True section")
                    target_q[field][idx][0][action_batch[idx][field]] = reward_batch[idx]
        training_history = self.q_net.fit(x=state_batch, y=target_q)
        loss = training_history.history['loss']
        return loss

    def build_dqn_model(self, input_array):
        input = tf.keras.Input(shape=input_array, dtype=np.float32)

        x = tf.keras.layers.Dense(256, activation='relu')(input)

        x = tf.keras.layers.Dense(128, activation='relu')(x)

        shop_output = tf.keras.layers.Dense(12, activation='linear')(x)
        item_output = tf.keras.layers.Dense(10, activation='linear')(x)
        bench_output = tf.keras.layers.Dense(9, activation='linear')(x)
        board_output_x = tf.keras.layers.Dense(7, activation='linear')(x)
        board_output_y = tf.keras.layers.Dense(4, activation='linear')(x)
        output = tf.keras.Model(inputs=input,
                                outputs=[shop_output, item_output, bench_output, board_output_x, board_output_y])
        output.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss='mse')
        return output
