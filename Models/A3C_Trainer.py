import config
import tensorflow as tf
import numpy as np
from datetime import datetime

class Trainer(object):
    def __init__(self):
        self.optimizer = tf.optimizers.RMSprop(learning_rate=1e-3)
        self.action_dim = config.ACTION_8D_DIM
        self.sequence_length = config.A3C_SEQUENCE_LENGTH
        self.gamma = 0.99
        self.batch_size = config.BATCH_SIZE

    def train_step(self, batch, agent, episode=0):
        # Trains the underlying network with batches of gameplay experience
        # Returns a training loss
        # obs_batch, logit_batch, action_batch, reward_batch, prev_action_batch = batch
        # (64, 2)    (64, 4)  (64, 5)     (64, 5)      (64, 5)     (64, 5) (64, 5), (64, 5,10)
        observation, history, value_mask, reward_mask, policy_mask, value, reward,  policy   , prev_action = batch
        # vr_s, _, _, vr_r = sample_sequence

        reward_batch = self.process_rewards(reward_batch, episode)
        # No need to process prev rewards because it wants the raw reward value rather the adjusted one
        # Since the model normally gets the raw reward value
        # sample_batch = self.process_rewards(vr_r, episode)

        act_re_input = []
        for i in range(self.sequence_length):
            act_re_input.append(agent.action_one_hot(prev_action_batch[:, i]))
        act_re_input = np.swapaxes(np.asarray(act_re_input), 0, 1)

        with tf.GradientTape() as tape:

            # Create gradients for the batch as a whole
            # obs_input = tf.convert_to_tensor(obs_batch, dtype=tf.float32)
            policy, norm_value, un_norm_value = [], [], []
            # state shape = (batch size, sequence length, 1, state dimensionality)
            # Moving over the sequence length because I need an output at each time step
            # and the action and policy do not have a sequence length component to them.
            for i in range(self.sequence_length):
                # I need to add the previous action and reward here.
                p, [norm_v, un_norm_v] = agent.a3c_net([observation[0], observation[1], prev_action], training=True)
                policy.append(self.transpose(p))
                norm_value.append(norm_v)
                un_norm_value.append(un_norm_v)
            norm_v = tf.convert_to_tensor(norm_value)
            un_norm_v = tf.convert_to_tensor(un_norm_value)
            v = tf.reshape(norm_v, (self.sequence_length, self.batch_size,))
            un_norm_v = tf.reshape(un_norm_v, (self.sequence_length, self.batch_size,))
            bootstrap = un_norm_v[-1]
            policy = self.transpose(policy)
            logit_batch = self.transpose(logit_batch)
            # policy, logit_batch = self.maskInput(policy, logit_batch, action_batch)

            # Create gradients for the random sub-sequence
            # vr_s_input = tf.convert_to_tensor(vr_s, dtype=tf.float32)
            # _, [_, v_vr] = self.a3c_net(vr_s_input, training=True)

            vs, advantage = self.from_logits(logit_batch, policy, action_batch, reward_batch, v, un_norm_v, bootstrap)

            normalized_vtrace = (vs - agent.popArtLayer.mean) / agent.popArtLayer.std

            normalized_vtrace = tf.nest.map_structure(tf.stop_gradient, normalized_vtrace)
            # Generate the losses
            a_loss = self.actor_loss(policy, action_batch, advantage)
            # Mean feels like it is more correct than sum here
            # Due to the numbers that sum produces vs the numbers that mean produces
            c_loss = 0.5 * tf.math.reduce_sum(tf.math.square(normalized_vtrace - v))
            # vr_loss = tf.math.reduce_mean(tf.keras.losses.mean_squared_error(sample_batch, v_vr))
            ac_loss = a_loss + 0.5 * c_loss

        ac_loss = tf.convert_to_tensor(ac_loss)
        grads = tape.gradient(ac_loss, agent.a3c_net.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(grads, 40)
        self.optimizer.apply_gradients(zip(grads, agent.a3c_net.trainable_variables))
        agent.popArtLayer.update_moments(vs)
        with self.file_writer.as_default():
            tf.summary.scalar('action loss', a_loss, episode)
            tf.summary.scalar('critic loss', c_loss, episode)
            # tf.summary.scalar('value replay loss', vr_loss, episode)
            tf.summary.histogram('final state value', tf.squeeze(bootstrap), episode)
        return ac_loss

    def process_rewards(self, rewards, episode=0):
        discnt_rewards = []
        sum_reward = 0
        rewards = np.flip(rewards)
        # self.t_board.on_train_batch_end(episode, {'reward': rewards[0]})
        for r in rewards:
            sum_reward = r + self.gamma * sum_reward
            discnt_rewards.append(sum_reward)
        discnt_rewards.reverse()
        discnt_rewards = np.array(discnt_rewards, dtype=np.float32)
        discnt_rewards = np.clip(discnt_rewards, -1.0, 1.0)
        return discnt_rewards

    def compute_entropy_loss(self, logits):
        entropy = []
        for i in range(len(self.action_dim)):
            logit = logits[i]
            policy = tf.nn.softmax(logit)
            log_policy = tf.nn.log_softmax(logit)
            entropy_per_timestep = tf.reduce_sum(-policy * log_policy, axis=-1)
            entropy.append(entropy_per_timestep)
        entropy = sum(entropy)
        return -tf.reduce_sum(entropy)

    def compute_policy_gradient_loss(self, logits, actions, advantages):
        target_action_log_probs = []
        for i in range(len(self.action_dim)):
            logit = logits[i]
            label = tf.transpose(tf.convert_to_tensor(actions[:, i], dtype=tf.int32), [1, 0])
            t_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=label)

            target_action_log_probs.append(t_loss)

        target_action_log_probs = sum(target_action_log_probs)
        advantages = tf.stop_gradient(advantages)
        policy_gradient_loss_per_timestep = target_action_log_probs * advantages
        return tf.reduce_sum(policy_gradient_loss_per_timestep)

    def actor_loss(self, probs, actions, advantage):
        total_loss = self.compute_policy_gradient_loss(probs, actions, advantage)
        total_loss += 0.001 * self.compute_entropy_loss(probs)
        return total_loss

    # I need to correct all of this to be time by batch instead of batch by time.
    def from_logits(self, behaviour_policy_logits, target_policy_logits, actions, rewards, values, un_norm_values,
                    bootstrap_value, clip_rho_threshold=1.0):
        target_action_log_probs = []
        for i in range(len(self.action_dim)):
            logit = tf.convert_to_tensor(target_policy_logits[:][:][i])
            label = tf.convert_to_tensor(actions[:, :, i], dtype=tf.int32)
            t_loss = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=label)

            target_action_log_probs.append(t_loss)

        target_action_log_probs = sum(target_action_log_probs)
        behaviour_action_log_probs = []

        # over each step
        for i in range(len(self.action_dim)):
            logit = behaviour_policy_logits[i]
            logit = tf.squeeze(tf.convert_to_tensor(logit, dtype=tf.float32))
            label = tf.convert_to_tensor(actions[:, i], dtype=tf.int32)
            b_loss = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=label)
            behaviour_action_log_probs.append(b_loss)
        behaviour_action_log_probs = sum(behaviour_action_log_probs)
        log_rhos = target_action_log_probs - self.transpose(behaviour_action_log_probs)
        vs, pg_advantages = self.from_importance_weights(
            log_rhos=log_rhos,
            rewards=rewards,
            values=values,
            un_norm_values=un_norm_values,
            bootstrap_value=bootstrap_value,
            clip_rho_threshold=clip_rho_threshold)
        return vs, pg_advantages

    def from_importance_weights(self, log_rhos, rewards, values, un_norm_values, bootstrap_value,
                                clip_rho_threshold=1.0):

        discounts = tf.fill([self.sequence_length - 1, self.batch_size], self.gamma)

        end_discount = tf.fill([1, self.batch_size], 0.0)
        discounts = tf.concat([discounts, end_discount], axis=0)
        rhos = tf.exp(log_rhos)
        clipped_rhos = tf.minimum(clip_rho_threshold, rhos, name='clipped_rhos')

        cs = tf.squeeze(rhos, name='cs')
        values_t_plus_1 = tf.concat([un_norm_values[1:], tf.expand_dims(bootstrap_value, 0)], axis=0)

        deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - un_norm_values)

        sequences = (discounts, cs, deltas)

        def scanfunc(acc, sequence_item):
            discount_t, c_t, delta_t = sequence_item
            return delta_t + discount_t * c_t * acc

        initial_values = tf.zeros_like(bootstrap_value)
        vs_minus_v_xs = tf.nest.map_structure(tf.stop_gradient, tf.scan(
            fn=scanfunc, elems=sequences, initializer=initial_values, parallel_iterations=1, reverse=True,))

        vs = tf.math.add_n([vs_minus_v_xs, un_norm_values])

        # Advantage for policy gradient.
        vs_t_plus_1 = tf.concat([vs[1:], tf.expand_dims(bootstrap_value, 0)], axis=0)
        # clipped_pg_rhos = tf.minimum(clip_pg_rho_threshold, rhos, name='clipped_pg_rhos')

        pg_advantages = clipped_rhos * (((rewards + self.gamma * vs_t_plus_1 - self.popArtLayer.mean.numpy())
                                         / self.popArtLayer.std.numpy()) - values)

        # Make sure no gradients backpropagated through the returned values.
        return tf.stop_gradient(vs), tf.stop_gradient(pg_advantages)

    def transpose(self, matrix):
        rows = len(matrix)
        columns = len(matrix[0])
        matrix_T = []
        for j in range(columns):
            row = []
            for i in range(rows):
                row.append(matrix[i][j])
            matrix_T.append(row)

        return matrix_T

    def maskInput(self, policy, logits, actions):
        for b in range(self.batch_size):
            for s in range(self.sequence_length):
                if actions[b, 0, s] < 8 or actions[b, 0, s] > 8:
                    one_hot_action = tf.one_hot(actions[b, 0, s], self.action_dim[1])
                    logits[1][b][s] = tf.expand_dims(one_hot_action, axis=0)
                if actions[b, 0, s] < 9 or actions[b, 0, s] == 11:
                    one_hot_action = tf.one_hot(actions[b, 0, s], self.action_dim[2])
                    logits[2][b][s] = tf.expand_dims(one_hot_action, axis=0)
                if actions[b, 0, s] < 8 or actions[b, 0, s] == 9:
                    one_hot_action = tf.one_hot(actions[b, 0, s], self.action_dim[3])
                    logits[3][b][s] = tf.expand_dims(one_hot_action, axis=0)
                    one_hot_action = tf.one_hot(actions[b, 0, s], self.action_dim[4])
                    logits[4][b][s] = tf.expand_dims(one_hot_action, axis=0)
        return policy, logits
