import config
import collections
import torch
import torch.nn.functional as F
import numpy as np
from torch.nn import L1Loss
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler

Prediction = collections.namedtuple(
    'Prediction',
    'value value_logits reward reward_logits policy_logits')


class Trainer(object):
    def __init__(self, global_agent):
        self.global_agent = global_agent
        self.optimizer = self.create_optimizer()
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        # self.value_loss = tf.keras.losses.MeanSquaredError()

    def create_optimizer(self):
        optimizer = torch.optim.SGD(self.global_agent.parameters(), lr=config.SGD_LR_INIT, 
                                    momentum=config.MOMENTUM, weight_decay=config.SGD_WEIGHT_DECAY)
        return optimizer
    
    # Same as muzero-general
    def adjust_lr(self, train_step):
        # adjust learning rate, step lr every lr_decay_steps
        lr = config.SGD_LR_INIT * config.SGD_LR_DECAY_RATE ** (
            train_step / config.SGD_LR_DECAY_STEPS
        )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def train_network(self, batch, agent, train_step, summary_writer):
        observation, history, value_mask, reward_mask, policy_mask, value, reward, policy = batch
        # if i % config.checkpoint_interval == 0:
        #     storage.save_network(i, network)
        # with tf.GradientTape() as tape:
        #     loss = self.compute_loss(agent, observation, history, value_mask, reward_mask, policy_mask,
        #                              value, reward, policy, train_step, summary_writer)

        # grads = tape.gradient(loss, agent.get_rl_training_variables())

        # self.optimizer.apply_gradients(zip(grads, agent.get_rl_training_variables()))

        # loss = self.compute_loss(...)
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        self.adjust_lr(train_step)
        self.optimizer.zero_grad()

        loss = self.compute_loss(agent, observation, history, value_mask, reward_mask, policy_mask,
                                 value, reward, policy, train_step, summary_writer)
        # loss.backward()
        # Apparently this works? i don't know if this is supposed to be mean or sum
        loss.mean().backward()
        self.optimizer.step()

        # storage.save_network(config.training_steps, network)\

    def compute_loss(self, agent, observation, history, target_value_mask, target_reward_mask, target_policy_mask,
                     target_value, target_reward, target_policy, train_step, summary_writer):

        target_value_mask = torch.from_numpy(target_value_mask)
        target_reward_mask = torch.from_numpy(target_reward_mask)
        target_policy_mask = torch.from_numpy(target_policy_mask)
        target_reward = torch.from_numpy(target_reward)
        target_value = torch.from_numpy(target_value)

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
            # 'action': target_policy_mask,
        }
        def name_to_mask(name):
            return next(k for k in masks if k in name)

        # This is more rigorous than the MuZero paper.
        gradient_scales = {
            k: torch.div(torch.tensor(1.0), torch.maximum(torch.sum(m[:, 1:], -1), torch.tensor(1)))
            for k, m in masks.items()
        }
        gradient_scales = {
            k: [torch.ones_like(s)] + [s] * (num_target_steps - 1)
            for k, s in gradient_scales.items()
        }

        target_reward_encoded, target_value_encoded = (torch.reshape(
            torch.from_numpy(enc.encode(torch.reshape(v, (-1,)))),
            (-1, num_target_steps,
             int(enc.num_steps))) for enc, v in ((agent.reward_encoder, target_reward),
                                            (agent.value_encoder, target_value)))

        accs = collections.defaultdict(list)
        for tstep, prediction in enumerate(predictions):
            # prediction.value_logits is [batch_size, 601]

            # TODO: Possibly keep them as tensors in the inference functions
            value = torch.from_numpy(prediction.value)
            reward = torch.from_numpy(prediction.reward)
            value_logits = torch.from_numpy(prediction.value_logits)
            reward_logits = torch.from_numpy(prediction.reward_logits)
            policy_logits = prediction.policy_logits if torch.is_tensor(prediction.policy_logits) else torch.tensor(prediction.policy_logits)

            accs['value_loss'].append(
                self.scale_gradient(self.loss_fct(value_logits,target_value_encoded[:, tstep]),
                                    gradient_scales['value'][tstep])
            )
            accs['reward_loss'].append(
                self.scale_gradient(self.loss_fct(reward_logits,target_reward_encoded[:, tstep]),
                                    gradient_scales['value'][tstep])
            )

            # predictions.policy_logits is (actiondims, batch) 
            # target_policy is (batch,unrollsteps+1,action_dims)

            # future ticket
            # entropy_loss = -tfd.Independent(tfd.Categorical(
            #     logits = logits, dtype=float), reinterpreted_batch_ndims=1).entropy()
            #     * config.policy_loss_entropy_regularizer
            policy_loss = self.loss_fct(policy_logits, torch.tensor([i[tstep] for i in target_policy]))
            # policy_loss = tf.reduce_sum(-tf.convert_to_tensor([i[tstep] for i in target_policy]) *
            #                             tf.nn.log_softmax(logits=prediction.policy_logits), -1)

            accs['policy_loss'].append(
                self.scale_gradient(policy_loss, gradient_scales['policy'][tstep]))

            accs['value_diff'].append(
                torch.abs(torch.squeeze(value) - target_value[:, tstep]))
            accs['reward_diff'].append(
                torch.abs(torch.squeeze(reward) - target_reward[:, tstep]))
            # accs['policy_acc'].append(
            #     tf.keras.metrics.categorical_accuracy(
            #         target_policy[:, tstep],
            #         tf.nn.softmax(prediction.policy_logits, axis=-1)))

            accs['value'].append(torch.squeeze(value))
            accs['reward'].append(torch.squeeze(reward))
            # accs['action'].append(
            #     tf.cast(tf.argmax(prediction.policy_logits, -1), tf.float32))

            accs['target_value'].append(target_value[:, tstep])
            accs['target_reward'].append(target_reward[:, tstep])
            # accs['target_action'].append(
            #     tf.cast(tf.argmax(target_policy[:, tstep], -1), tf.float32))

        accs = {k: torch.stack(v, -1) * masks[name_to_mask(k)] for k, v in accs.items()}

        loss = accs['value_loss'] + config.REWARD_LOSS_SCALING * accs[
            'reward_loss'] + config.POLICY_LOSS_SCALING * accs['policy_loss']
        mean_loss = torch.sum(loss, -1).to('cuda')  # aggregating over time

        # Leaving this here in case I want to use it later.
        # This was used in Atari but not in board games. Also, very unclear how to
        # Create the importance_weights from paper or from the source code.
        # loss = loss * importance_weights  # importance sampling correction
        # mean_loss = tf.math.divide_no_nan(
        #     tf.reduce_sum(loss), tf.reduce_sum(importance_weights))

        if config.WEIGHT_DECAY > 0.:
            l2_loss = config.WEIGHT_DECAY * sum(
                self.l2_loss(p)
                for p in agent.parameters())
        else:
            l2_loss = mean_loss * 0.

        mean_loss += l2_loss

        sum_accs = {k: torch.sum(a, -1) for k, a in accs.items()}
        sum_masks = {
            k: torch.maximum(torch.sum(m, -1), torch.tensor(1.)) for k, m in masks.items()
        }

        def get_mean(k):
            return torch.mean(sum_accs[k] / sum_masks[name_to_mask(k)])

        summary_writer.add_scalar('prediction/value', get_mean('value'), train_step)
        summary_writer.add_scalar('prediction/reward', get_mean('reward'), train_step)

        summary_writer.add_scalar('target/value', get_mean('target_value'), train_step)
        summary_writer.add_scalar('target/reward', get_mean('target_reward'), train_step)

        summary_writer.add_scalar('losses/value', torch.mean(sum_accs['value_loss']), train_step)
        summary_writer.add_scalar('losses/reward', torch.mean(sum_accs['reward_loss']), train_step)
        summary_writer.add_scalar('losses/policy', torch.mean(sum_accs['policy_loss']), train_step)
        summary_writer.add_scalar('losses/total', torch.mean(mean_loss), train_step)
        summary_writer.add_scalar('losses/l2', l2_loss, train_step)

        summary_writer.add_scalar('accuracy/value', -get_mean('value_diff'), train_step)
        summary_writer.add_scalar('accuracy/reward', -get_mean('reward_diff'), train_step)

        summary_writer.add_scalar('episode_max/reward', torch.max(target_reward), train_step)
        summary_writer.add_scalar('episode_max/value', torch.max(target_value), train_step)

        return mean_loss

    def scale_gradient(self, t, scale):
        return scale * t + (1 - scale) * t.detach()

    def l2_loss(self, t):
        return torch.sum(t ** 2) / 2