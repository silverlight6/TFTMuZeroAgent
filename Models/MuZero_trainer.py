import config
import collections
import tensorflow as tf

Prediction = collections.namedtuple(
    'Prediction',
    'value value_logits reward reward_logits policy_logits')


class Trainer(object):
    def __init__(self):
        self.optimizer, self.learning_rate_fn = self.create_optimizer()
        self.value_loss = tf.keras.losses.MeanSquaredError()

    def create_optimizer(self):
        learning_rate_fn = tf.keras.experimental.CosineDecay(
            config.INIT_LEARNING_RATE,
            config.LEARNING_RATE_DECAY,
            alpha=config.LR_DECAY_FUNCTION)
        optimizer = tf.keras.optimizers.Adam(learning_rate_fn)
        return optimizer, learning_rate_fn

    def train_network(self, batch, agent, train_step, summary_writer):
        observation, history, value_mask, reward_mask, policy_mask, value, reward, policy = batch
        # if i % config.checkpoint_interval == 0:
        #     storage.save_network(i, network)
        with tf.GradientTape() as tape:
            loss = self.compute_loss(agent, observation, history, value_mask, reward_mask, policy_mask,
                                     value, reward, policy, train_step, summary_writer)

        grads = tape.gradient(loss, agent.get_rl_training_variables())

        self.optimizer.apply_gradients(zip(grads, agent.get_rl_training_variables()))
        # storage.save_network(config.training_steps, network)\

    def compute_loss(self, agent, observation, history, target_value_mask, target_reward_mask, target_policy_mask,
                     target_value, target_reward, target_policy, train_step, summary_writer):
        # initial step
        output = agent.initial_inference(observation)

        predictions = [
            Prediction(
                value=output["value"],
                value_logits=output["value_logits"],
                reward=output["reward"],
                reward_logits=output["reward_logits"],
                policy_logits=output["policy_logits"],
            )
        ]

        # recurrent steps
        num_recurrent_steps = config.UNROLL_STEPS
        for rstep in range(num_recurrent_steps):
            hidden_state_gradient_scale = 1.0 if rstep == 0 else 0.5
            output = agent.recurrent_inference(
                self.scale_gradient(output["hidden_state"], hidden_state_gradient_scale),
                history[:, rstep],
            )
            predictions.append(
                Prediction(
                    value=output["value"],
                    value_logits=output["value_logits"],
                    reward=output["reward"],
                    reward_logits=output["reward_logits"],
                    policy_logits=output["policy_logits"],
                ))

        num_target_steps = target_value.shape[-1]
        assert len(predictions) == num_target_steps, (
            'There should be as many predictions ({}) as targets ({})'.format(
                len(predictions), num_target_steps))

        masks = {
            'value': target_value_mask,
            'reward': target_reward_mask,
            'policy': target_policy_mask,
            'action': target_policy_mask,
        }

        def name_to_mask(name):
            return next(k for k in masks if k in name)

        # This is more rigorous than the MuZero paper.
        gradient_scales = {
            k: tf.math.divide(1.0, tf.maximum(tf.reduce_sum(m[:, 1:], -1), 1))
            for k, m in masks.items()
        }
        gradient_scales = {
            k: [tf.ones_like(s)] + [s] * (num_target_steps - 1)
            for k, s in gradient_scales.items()
        }

        target_reward_encoded, target_value_encoded = (tf.reshape(
            enc.encode(tf.reshape(v, (-1,))),
            (-1, num_target_steps,
             enc.num_steps)) for enc, v in ((agent.reward_encoder, target_reward),
                                            (agent.value_encoder, target_value)))

        accs = collections.defaultdict(list)
        for tstep, prediction in enumerate(predictions):
            # prediction.value_logits is [64, 601]
            accs['value_loss'].append(
                self.scale_gradient(tf.nn.softmax_cross_entropy_with_logits(
                    logits=prediction.value_logits,
                    labels=target_value_encoded[:, tstep]),
                    gradient_scales['value'][tstep]))
            accs['reward_loss'].append(
                self.scale_gradient(tf.nn.softmax_cross_entropy_with_logits(
                    logits=prediction.reward_logits,
                    labels=target_reward_encoded[:, tstep]),
                    gradient_scales['reward'][tstep]))

            # predictions.policy_logits is (actiondims, batch) 
            # target_policy is (batch,unrollsteps+1,action_dims)

            # future ticket
            # entropy_loss = -tfd.Independent(tfd.Categorical(
            #     logits = logits, dtype=float), reinterpreted_batch_ndims=1).entropy()
            #     * config.policy_loss_entropy_regularizer
            policy_loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=prediction.policy_logits, labels=tf.convert_to_tensor([i[tstep] for i in target_policy]))
            # policy_loss = tf.reduce_sum(-tf.convert_to_tensor([i[tstep] for i in target_policy]) *
            #                             tf.nn.log_softmax(logits=prediction.policy_logits), -1)

            accs['policy_loss'].append(
                self.scale_gradient(policy_loss, gradient_scales['policy'][tstep]))

            accs['value_diff'].append(
                tf.abs(tf.squeeze(prediction.value) - target_value[:, tstep]))
            accs['reward_diff'].append(
                tf.abs(tf.squeeze(prediction.reward) - target_reward[:, tstep]))
            # accs['policy_acc'].append(
            #     tf.keras.metrics.categorical_accuracy(
            #         target_policy[:, tstep],
            #         tf.nn.softmax(prediction.policy_logits, axis=-1)))

            accs['value'].append(tf.squeeze(prediction.value))
            accs['reward'].append(tf.squeeze(prediction.reward))
            # accs['action'].append(
            #     tf.cast(tf.argmax(prediction.policy_logits, -1), tf.float32))

            accs['target_value'].append(target_value[:, tstep])
            accs['target_reward'].append(target_reward[:, tstep])
            # accs['target_action'].append(
            #     tf.cast(tf.argmax(target_policy[:, tstep], -1), tf.float32))

        accs = {k: tf.stack(v, -1) * masks[name_to_mask(k)] for k, v in accs.items()}

        loss = accs['value_loss'] + config.REWARD_LOSS_SCALING * accs[
            'reward_loss'] + config.POLICY_LOSS_SCALING * accs['policy_loss']
        mean_loss = tf.reduce_sum(loss, -1)  # aggregating over time

        # Leaving this here in case I want to use it later.
        # This was used in Atari but not in board games. Also, very unclear how to
        # Create the importance_weights from paper or from the source code.
        # loss = loss * importance_weights  # importance sampling correction
        # mean_loss = tf.math.divide_no_nan(
        #     tf.reduce_sum(loss), tf.reduce_sum(importance_weights))

        if config.WEIGHT_DECAY > 0.:
            l2_loss = config.WEIGHT_DECAY * sum(
                tf.nn.l2_loss(v)
                for v in agent.get_rl_training_variables())
        else:
            l2_loss = mean_loss * 0.

        mean_loss += l2_loss

        sum_accs = {k: tf.reduce_sum(a, -1) for k, a in accs.items()}
        sum_masks = {
            k: tf.maximum(tf.reduce_sum(m, -1), 1.) for k, m in masks.items()
        }

        def get_mean(k):
            return tf.reduce_mean(sum_accs[k] / sum_masks[name_to_mask(k)])

        with summary_writer.as_default():
            # TODO(lobotuerk): Find a better way of logging difference between predictions and targets
            # Just the mean does not tell you a lot of information, you dont really know if you had outliers
            # or if you were off in all predictions. Properties (sd, variance) would be better
            tf.summary.scalar('prediction/value', get_mean('value'), step=train_step)
            tf.summary.scalar('prediction/reward', get_mean('reward'), step=train_step)

            tf.summary.scalar('target/value', get_mean('target_value'), step=train_step)
            tf.summary.scalar('target/reward', get_mean('target_reward'), step=train_step)

            tf.summary.scalar('losses/value', tf.reduce_mean(sum_accs['value_loss']), step=train_step)
            tf.summary.scalar('losses/reward', tf.reduce_mean(sum_accs['reward_loss']), step=train_step)
            tf.summary.scalar('losses/policy', tf.reduce_mean(sum_accs['policy_loss']), step=train_step)
            tf.summary.scalar('losses/total', tf.reduce_mean(mean_loss), step=train_step)
            tf.summary.scalar('losses/l2', l2_loss, step=train_step)

            tf.summary.scalar('accuracy/value', -get_mean('value_diff'), step=train_step)
            tf.summary.scalar('accuracy/reward', -get_mean('reward_diff'), step=train_step)

            tf.summary.scalar('episode_max/reward', tf.reduce_max(target_reward), step=train_step)
            tf.summary.scalar('episode_max/value', tf.reduce_max(target_value), step=train_step)

        return mean_loss

    def scale_gradient(self, t, scale):
        return scale * t + (1 - scale) * tf.stop_gradient(t)
