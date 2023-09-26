import config
import collections
import torch
import numpy as np
from Models.MCTS_Util import split_batch, map_output_to_distribution, action_to_idx

Prediction = collections.namedtuple(
    'Prediction',
    'value value_logits reward reward_logits policy_logits')


class Trainer(object):
    def __init__(self, global_agent):
        self.global_agent = global_agent
        self.init_learning_rate = config.INIT_LEARNING_RATE
        self.decay_steps = 1.0
        self.alpha = config.LR_DECAY_FUNCTION
        self.optimizer = self.create_optimizer()

    def create_optimizer(self):
        optimizer = torch.optim.Adam(self.global_agent.parameters(), lr=config.INIT_LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        return optimizer

    def decayed_learning_rate(self, step):
        step = min(step, self.decay_steps)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * step / self.decay_steps))
        decayed = (1 - self.alpha) * cosine_decay + self.alpha
        return self.init_learning_rate * decayed

    # Same as muzero-general
    def adjust_lr(self, train_step):
        lr = self.init_learning_rate

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def train_network(self, batch, agent, train_step, summary_writer):
        observation, history, value_mask, reward_mask, policy_mask, value, reward, policy, sample_set = batch
        agent.train()
        self.adjust_lr(train_step)

        self.optimizer.zero_grad()

        sample_set, policy = split_batch(sample_set, policy)  # [unroll_steps, num_dims, [(batch_size, dim) ...] ]

        loss = self.compute_loss(agent, observation, history, value_mask, reward_mask, policy_mask,
                                 value, reward, policy, sample_set, train_step, summary_writer)

        loss = loss.sum()

        loss.backward()

        self.optimizer.step()

    def compute_loss(self, agent, observation, history, target_value_mask, target_reward_mask, target_policy_mask,
                     target_value, target_reward, target_policy, sample_set, train_step, summary_writer):

        # target_value_mask = torch.from_numpy(target_value_mask).to('cuda')
        # target_reward_mask = torch.from_numpy(target_reward_mask).to('cuda')
        # target_policy_mask = torch.from_numpy(target_policy_mask).to('cuda')
        target_reward = torch.from_numpy(target_reward).to('cuda')
        target_value = torch.from_numpy(target_value).to('cuda')

        # initial step
        output = agent.initial_inference(observation)  # [num_dims, [(batch_size, dim) ...] ]

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
            # 1.0 and 0.5 because we scale prior to the recurrent inference. 0.5 if scaled afterwards.
            hidden_state_gradient_scale = 1.0 if rstep == 0 else 0.5
            hidden_state = output["hidden_state"]
            hidden_state.requires_grad_(True).register_hook(lambda grad: grad * hidden_state_gradient_scale)
            output = agent.recurrent_inference(
                hidden_state,
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

        # masks = {
        #     'value': target_value_mask,
        #     'reward': target_reward_mask,
        #     'policy': target_policy_mask,
        # }

        # def name_to_mask(name):
        #     return next(k for k in masks if k in name)

        target_reward_encoded, target_value_encoded = (torch.reshape(
            torch.tensor(enc.encode(torch.reshape(v, (-1,)).to('cpu'))).to('cuda'),
            (-1, num_target_steps,
             int(enc.num_steps))) for enc, v in ((agent.reward_encoder, target_reward),
                                                 (agent.value_encoder, target_value)))

        accs = collections.defaultdict(list)
        target_policy = torch.reshape(torch.tensor(np.array(target_policy)), (-1, num_target_steps, 728)).to('cuda').requires_grad_(True)
        for tstep, prediction in enumerate(predictions):
            # prediction.value_logits is [batch_size, 601]

            # TODO: Possibly keep them as tensors in the inference functions
            value = torch.from_numpy(prediction.value).to('cuda')
            # print("VALUE", max(value).item(), min(value).item(), (max(value) - min(value)).item())
            reward = torch.from_numpy(prediction.reward).to('cuda')
            value_logits = prediction.value_logits.to('cuda').requires_grad_(True)
            reward_logits = prediction.reward_logits.to('cuda') if torch.is_tensor(prediction.reward_logits) \
                else torch.tensor(prediction.reward_logits).to('cuda')
            reward_logits = reward_logits.requires_grad_(True)
            policy_logits = prediction.policy_logits[0].to('cuda').requires_grad_(True)

            cross_loss = torch.nn.CrossEntropyLoss(reduction = 'sum')

            value_loss = cross_loss(value_logits, target_value_encoded[:, tstep])
            value_loss.register_hook(lambda grad: grad / config.UNROLL_STEPS)

            accs['value_loss'].append(
                value_loss
            )

            reward_loss = cross_loss(reward_logits, target_reward_encoded[:, tstep])
            reward_loss.register_hook(lambda grad: grad / config.UNROLL_STEPS)

            accs['reward_loss'].append(
                reward_loss
            )

            # future ticket
            # entropy_loss = -tfd.Independent(tfd.Categorical(
            #     logits = logits, dtype=float), reinterpreted_batch_ndims=1).entropy()
            #     * config.policy_loss_entropy_regularizer

            # predictions.policy_logits is (actiondims, batch)
            # target_policy is (batch, unrollsteps+1, action_dims)

            # target_policy -> [ [(256, 7), (256, n), ...] * tstep ]
            # output_policy ->   [(256, 7), (256, n), ...]
            policy_loss = cross_loss(policy_logits, target_policy[:, tstep])

            policy_loss.register_hook(lambda grad: grad / config.UNROLL_STEPS)

            accs['policy_loss'].append(policy_loss)

            accs['value_diff'].append(torch.abs(torch.squeeze(value) - target_value[:, tstep]))
            accs['reward_diff'].append(torch.abs(torch.squeeze(reward) - target_reward[:, tstep]))

            accs['value'].append(torch.squeeze(value))
            accs['reward'].append(torch.squeeze(reward))

            accs['target_value'].append(target_value[:, tstep])
            accs['target_reward'].append(target_reward[:, tstep])

        accs = {k: torch.stack(v, -1) for k, v in accs.items()}

        loss = accs['value_loss'] + config.REWARD_LOSS_SCALING * accs[
            'reward_loss'] + config.POLICY_LOSS_SCALING * accs['policy_loss']
        mean_loss = torch.sum(loss, -1).to('cuda')  # aggregating over time

        # Leaving this here in case I want to use it later.
        # This was used in Atari but not in board games. Also, very unclear how to
        # Create the importance_weights from paper or from the source code.
        # loss = loss * importance_weights  # importance sampling correction
        # mean_loss = tf.math.divide_no_nan(
        #     tf.reduce_sum(loss), tf.reduce_sum(importance_weights))

        # if config.WEIGHT_DECAY > 0.:
        #     l2_loss = config.WEIGHT_DECAY * sum(
        #         self.l2_loss(p)
        #         for p in agent.parameters())
        # else:
        #     l2_loss = mean_loss * 0.

        # mean_loss += l2_loss

        sum_accs = {k: torch.sum(a, -1) for k, a in accs.items()}
        # sum_masks = {
        #     k: torch.maximum(torch.sum(m, -1), torch.tensor(1.)) for k, m in masks.items()
        # }

        def get_mean(k):
            return torch.mean(sum_accs[k])

        summary_writer.add_scalar('prediction/value', get_mean('value'), train_step)
        summary_writer.add_scalar('prediction/reward', get_mean('reward'), train_step)

        summary_writer.add_scalar('target/value', get_mean('target_value'), train_step)
        summary_writer.add_scalar('target/reward', get_mean('target_reward'), train_step)

        summary_writer.add_scalar('losses/value', torch.mean(sum_accs['value_loss'] / config.UNROLL_STEPS), train_step)
        summary_writer.add_scalar('losses/reward', torch.mean(sum_accs['reward_loss'] / config.UNROLL_STEPS), train_step)
        summary_writer.add_scalar('losses/policy', torch.mean(sum_accs['policy_loss'] / config.UNROLL_STEPS), train_step)
        summary_writer.add_scalar('losses/total', torch.mean(mean_loss), train_step)
        # summary_writer.add_scalar('losses/l2', l2_loss, train_step)

        summary_writer.add_scalar('accuracy/value', -get_mean('value_diff'), train_step)
        summary_writer.add_scalar('accuracy/reward', -get_mean('reward_diff'), train_step)

        summary_writer.add_scalar('episode_max/reward', torch.max(target_reward), train_step)
        summary_writer.add_scalar('episode_max/value', torch.max(target_value), train_step)
        summary_writer.flush()

        return mean_loss

    def scale_gradient(self, grad, scale):
        return scale * grad

    def l2_loss(self, t):
        return torch.sum(t ** 2) / 2
