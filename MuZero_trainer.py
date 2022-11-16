import config
import collections
import tensorflow as tf

Prediction = collections.namedtuple(
    'Prediction',
    'value value_logits reward reward_logits policy_logits')


class Trainer(object):
    def __init__(self):
        self.optimizer, self.learning_rate_fn = self.create_optimizer()

    def create_optimizer(self):
        learning_rate_fn = tf.keras.experimental.CosineDecay(
            config.INIT_LEARNING_RATE,
            config.LEARNING_RATE_DECAY,
            alpha=config.LR_DECAY_FUNCTION)
        optimizer = tf.keras.optimizers.Adam(learning_rate_fn)
        return optimizer, learning_rate_fn

    def train_network(self, batch, agent):
        observation, history, value, reward, policy = batch
        # if i % config.checkpoint_interval == 0:
        #     storage.save_network(i, network)
        with tf.GradientTape() as tape:
            loss = self.compute_loss(agent, observation, history, value, reward, policy)

        grads = tape.gradient(loss, agent.get_trainable_variables())
        self.optimizer.apply_gradients(grads, agent.get_trainable_variables())
        # storage.save_network(config.training_steps, network)

    def compute_loss(self, agent, observation, history, target_value, target_reward, target_policy):

        # initial step
        output = agent.initial_inference(observation)
        predictions = [
            Prediction(
                value=output.value,
                value_logits=output.value_logits,
                reward=output.reward,
                reward_logits=output.reward_logits,
                policy_logits=output.policy_logits,
            )
        ]

        # recurrent steps
        num_recurrent_steps = history.shape[-1]
        for rstep in range(num_recurrent_steps):
            hidden_state_gradient_scale = 1.0 if rstep == 0 else 0.5
            output = agent.recurrent_inference(
                self.scale_gradient(output.hidden_state, hidden_state_gradient_scale),
                history[:, rstep],
            )
            predictions.append(
                Prediction(
                    value=output.value,
                    value_logits=output.value_logits,
                    reward=output.reward,
                    reward_logits=output.reward_logits,
                    policy_logits=output.policy_logits,
                ))

        num_target_steps = target_value.shape[-1]
        assert len(predictions) == num_target_steps, (
            'There should be as many predictions ({}) as targets ({})'.format(
                len(predictions), num_target_steps))

        target_reward_encoded, target_value_encoded = (tf.reshape(
            enc.encode(tf.reshape(v, (-1,))),
            (-1, num_target_steps,
             enc.num_steps)) for enc, v in ((agent.reward_encoder, target_reward),
                                            (agent.value_encoder, target_value)))

        # Accumulators over time steps.
        accs = collections.defaultdict(list)
        for tstep, prediction in enumerate(predictions):
            accs['value_loss'].append(
                self.scale_gradient(
                    tf.nn.softmax_cross_entropy_with_logits(
                        logits=prediction.value_logits,
                        labels=target_value_encoded[:, tstep]),
                    1 / num_recurrent_steps))
            accs['reward_loss'].append(
                self.scale_gradient(
                    tf.nn.softmax_cross_entropy_with_logits(
                        logits=prediction.reward_logits,
                        labels=target_reward_encoded[:, tstep]),
                    1 / num_recurrent_steps))
            policy_loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=prediction.policy_logits, labels=target_policy[:, tstep])
            # entropy_loss = -parametric_action_distribution.entropy(
            #     prediction.policy_logits) * config.policy_loss_entropy_regularizer
            #### policy_loss + entropy_loss if adding in entropy_loss.
            accs['policy_loss'].append(
                self.scale_gradient(policy_loss, 1 / num_recurrent_steps))

            # accs['value_diff'].append(
            #     tf.abs(tf.squeeze(prediction.value) - target_value[:, tstep]))
            # accs['reward_diff'].append(
            #     tf.abs(tf.squeeze(prediction.reward) - target_reward[:, tstep]))
            # accs['policy_acc'].append(
            #     tf.keras.metrics.categorical_accuracy(
            #         target_policy[:, tstep],
            #         tf.nn.softmax(prediction.policy_logits, axis=-1)))
            #
            # accs['value'].append(tf.squeeze(prediction.value))
            # accs['reward'].append(tf.squeeze(prediction.reward))
            # accs['action'].append(
            #     tf.cast(tf.argmax(prediction.policy_logits, -1), tf.float32))
            #
            # accs['target_value'].append(target_value[:, tstep])
            # accs['target_reward'].append(target_reward[:, tstep])
            # accs['target_action'].append(
            #     tf.cast(tf.argmax(target_policy[:, tstep], -1), tf.float32))

        # if config.debug and np.random.rand() < 1 / 50:
        #     logging.info('-------------------')
        #     logging.info(observation)
        #     for k, v in accs.items():
        #         logging.info('{}:\n{}'.format(k, v))
        #     for k, v in masks.items():
        #         logging.info('mask {}:\n{}'.format(k, v))
        #     logging.info('history:\n{}'.format(history))
        #     logging.info('target_policy:\n{}'.format(target_policy))
        #     logging.info('importance_weights:\n{}'.format(importance_weights))
        #     logging.info('-------------------')

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
                for v in agent.get_trainable_variables(pretraining=False))
        else:
            l2_loss = mean_loss * 0.

        mean_loss += l2_loss

        # log('losses/total', mean_loss)
        # log('losses/weight_decay', l2_loss)

        # sum_accs = {k: tf.reduce_sum(a, -1) for k, a in accs.items()}
        # sum_masks = {
        #     k: tf.maximum(tf.reduce_sum(m, -1), 1.) for k, m in masks.items()
        # }

        # def get_mean(k):
        #     return tf.reduce_mean(sum_accs[k] / sum_masks[name_to_mask(k)])

        # log('prediction/value', get_mean('value'))
        # log('prediction/reward', get_mean('reward'))
        # log('prediction/policy', get_mean('action'))
        #
        # log('target/value', get_mean('target_value'))
        # log('target/reward', get_mean('target_reward'))
        # log('target/policy', get_mean('target_action'))
        #
        # log('losses/value', tf.reduce_mean(sum_accs['value_loss']))
        # log('losses/reward', tf.reduce_mean(sum_accs['reward_loss']))
        # log('losses/policy', tf.reduce_mean(sum_accs['policy_loss']))
        #
        # log('accuracy/value', -get_mean('value_diff'))
        # log('accuracy/reward', -get_mean('reward_diff'))
        # log('accuracy/policy', get_mean('policy_acc'))

        return mean_loss

    def scale_gradient(self, t, scale):
        return scale * t + (1 - scale) * tf.stop_gradient(t)
