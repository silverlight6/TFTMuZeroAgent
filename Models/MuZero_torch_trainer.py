import config
import collections
import torch
import torch.nn.functional as F
import numpy as np
from Models.MCTS_Util import split_batch, sample_set_to_idx, create_target_and_mask

Prediction = collections.namedtuple(
    'Prediction',
    'value value_logits reward reward_logits policy_logits')

LossOutput = collections.namedtuple(
    'LossOutput',
    'value_loss reward_loss policy_loss value reward policy target_value target_reward l2_loss')


class Trainer(object):
    def __init__(self, global_agent, summary_writer):
        self.network = global_agent
        self.init_learning_rate = config.INIT_LEARNING_RATE
        self.decay_steps = config.WEIGHT_DECAY
        self.alpha = config.LR_DECAY_FUNCTION
        self.optimizer = self.create_optimizer()
        self.summary_writer = summary_writer

    def create_optimizer(self):
        optimizer = torch.optim.Adam(self.network.parameters(), lr=config.INIT_LEARNING_RATE,
                                     weight_decay=config.WEIGHT_DECAY)
        return optimizer

    def decayed_learning_rate(self, step):
        step = min(step, self.decay_steps)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * step / self.decay_steps))
        decayed = (1 - self.alpha) * cosine_decay + self.alpha
        return self.init_learning_rate * decayed

    # Same as muzero-general
    def adjust_lr(self, train_step):
        lr = self.decayed_learning_rate(train_step)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def train_network(self, batch, train_step):
        observation, action_history, value_mask, reward_mask, policy_mask, target_value, target_reward, target_policy, sample_set = batch
        self.adjust_lr(train_step)

        predictions = self.compute_forward(observation, action_history)

        sample_set, target_policy = split_batch(sample_set, target_policy)

        self.compute_loss(predictions, target_value, target_reward, target_policy, sample_set,
                          value_mask, reward_mask, policy_mask)

        self.backpropagate()

        self.write_summaries(train_step)

    def compute_forward(self, observation, action_history):
        self.network.train()
        grad_scale = 0.5

        output = self.network.initial_inference(observation)

        predictions = [
            Prediction(
                value=output["value"],
                value_logits=output["value_logits"],
                # The reward logits in initial inference are not from the network,
                # and might not be on the correct device
                reward=output["reward"].to(config.DEVICE),
                reward_logits=output["reward_logits"].to(config.DEVICE),
                policy_logits=output["policy_logits"],
            )
        ]

        for unroll_step in range(config.UNROLL_STEPS):
            hidden_state = output["hidden_state"]
            output = self.network.recurrent_inference(
                hidden_state, action_history[:, unroll_step])

            scale_gradient(hidden_state, grad_scale)

            predictions.append(
                Prediction(
                    value=output["value"],
                    value_logits=output["value_logits"],
                    reward=output["reward"],
                    reward_logits=output["reward_logits"],
                    policy_logits=output["policy_logits"],
                )
            )

        return predictions

    def compute_loss(self, predictions, target_value, target_reward, target_policy, sample_set,
                     value_mask, reward_mask, policy_mask):
        value_mask = torch.from_numpy(value_mask).to(config.DEVICE)
        reward_mask = torch.from_numpy(reward_mask).to(config.DEVICE)
        policy_mask = torch.from_numpy(policy_mask).to(config.DEVICE)

        target_value = self.encode_target(
            target_value, self.network.value_encoder).to(config.DEVICE)
        target_reward = self.encode_target(
            target_reward, self.network.reward_encoder).to(config.DEVICE)

        self.outputs = LossOutput(
            value_loss=[],
            reward_loss=[],
            policy_loss=[],
            value=[],
            reward=[],
            policy=[],
            target_value=[],
            target_reward=[],
            l2_loss=[]
        )

        for tstep, prediction in enumerate(predictions):
            step_value, step_target_value = prediction.value_logits, target_value[:, tstep]
            value_loss = self.value_or_reward_loss(step_value, step_target_value)
            self.scale_loss(value_loss)

            step_reward, step_target_reward = prediction.reward_logits, target_reward[:, tstep]
            reward_loss = self.value_or_reward_loss(step_reward, step_target_reward)
            self.scale_loss(reward_loss)

            step_policy, step_target_policy = self.mask_and_fill_policy(
                prediction.policy_logits, target_policy[tstep], sample_set[tstep])
            policy_loss = self.policy_loss(step_policy, step_target_policy)
            self.scale_loss(policy_loss)

            self.outputs.value_loss.append(value_loss)
            self.outputs.reward_loss.append(reward_loss)
            self.outputs.policy_loss.append(policy_loss)

            self.outputs.value.append(prediction.value)
            self.outputs.reward.append(prediction.reward)
            self.outputs.policy.append(prediction.policy_logits)

            self.outputs.target_value.append(self.decode_target(step_target_value, self.network.value_encoder))
            self.outputs.target_reward.append(self.decode_target(step_target_value, self.network.value_encoder))

        l2_loss = self.l2_regularization()
        self.outputs.l2_loss.append(l2_loss)

        value_loss = torch.stack(self.outputs.value_loss, -1) * value_mask
        reward_loss = torch.stack(self.outputs.reward_loss, -1) * reward_mask
        policy_loss = torch.stack(self.outputs.policy_loss, -1) * policy_mask

        self.loss = torch.sum(
            value_loss + reward_loss * config.REWARD_LOSS_SCALING +
            policy_loss * config.POLICY_LOSS_SCALING, -1).to(config.DEVICE)

        self.loss += l2_loss

        self.loss = self.loss.mean()

    def backpropagate(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def write_summaries(self, train_step):
        self.summary_writer.add_scalar('losses/total', self.loss, train_step)

        self.summary_writer.add_scalar(
            'losses/value', torch.mean(torch.stack(self.outputs.value_loss)), train_step)
        self.summary_writer.add_scalar(
            'losses/reward', torch.mean(torch.stack(self.outputs.reward_loss)), train_step)
        self.summary_writer.add_scalar(
            'losses/policy', torch.mean(torch.stack(self.outputs.policy_loss)), train_step)
        self.summary_writer.add_scalar(
            'losses/l2', torch.mean(torch.stack(self.outputs.l2_loss)), train_step)

        self.summary_writer.add_scalar(
            'prediction/value', torch.mean(torch.stack(self.outputs.value)), train_step)
        self.summary_writer.add_scalar(
            'prediction/reward', torch.mean(torch.stack(self.outputs.reward)), train_step)

        self.summary_writer.add_scalar(
            'target/value', torch.mean(torch.stack(self.outputs.target_value)), train_step)
        self.summary_writer.add_scalar(
            'target/reward', torch.mean(torch.stack(self.outputs.target_reward)), train_step)

        self.summary_writer.add_scalar(
            'episode_max/value', torch.max(torch.stack(self.outputs.target_value)), train_step)
        self.summary_writer.add_scalar(
            'episode_max/reward', torch.max(torch.stack(self.outputs.target_reward)), train_step)

        # for i in range(len(config.POLICY_HEAD_SIZES)):
        #     self.summary_writer.add_scalar(
        #         'episode_info/value_diff_{}'.format(i),
        #         torch.max(torch.max(torch.stack(self.outputs.policy[:, :, i]), 1).values -
        #                   torch.min(torch.stack(self.outputs.policy[:, :, i]), 1).values), train_step)

        self.summary_writer.flush()

    # Convert target from
    # [batch_size, unroll_steps]
    # to
    # [batch_size, unroll_steps, encoding_size]
    def encode_target(self, target, encoder):
        target = torch.from_numpy(target)
        target_flattened = torch.reshape(target, (-1,))
        target_encoded = encoder.encode(target_flattened)
        target_reshaped = torch.reshape(
            target_encoded,
            (-1, target.shape[-1], int(encoder.num_steps))
        )
        return target_reshaped

    def decode_target(self, target, decoder):
        target_flattened = torch.reshape(target, (target.shape[0], -1))
        target_encoded = decoder.decode(target_flattened)
        target_reshaped = torch.reshape(
            target_encoded,
            (-1, 1)
        )
        return target_reshaped

    # prediction [ [batch_size, action_dim_1], ...]
    # target [ [batch_size, sampled_action_dim_1], ...] (smaller than prediction)
    # sample_set [ [batch_size, sampled_action_dim_1], ...]
    # We need to mask the prediction so that only the sampled actions are used in the loss.
    # We also need to fill the target with zeros where the prediction is masked.
    def mask_and_fill_policy(self, prediction, target, sample_set):
        idx_set = sample_set_to_idx(sample_set)
        target, mask = create_target_and_mask(target, idx_set)

        # apply mask
        prediction = [pred_dim * torch.from_numpy(mask_dim).to(config.DEVICE)
                      for pred_dim, mask_dim in zip(prediction, mask)]

        target = [
            torch.from_numpy(target_dim).to(config.DEVICE) for target_dim in target
        ]

        return prediction, target

    def scale_loss(self, loss):
        scale_gradient(loss, 1.0 / config.UNROLL_STEPS)

    # Loss for each type of prediction
    def value_or_reward_loss(self, prediction, target):
        return cross_entropy_loss(prediction, target)

    def policy_loss(self, prediction, target):
        loss = 0.0
        for pred_dim, target_dim in zip(prediction, target):
            loss += cross_entropy_loss(pred_dim, target_dim)
        return loss

    def l2_regularization(self):
        return config.WEIGHT_DECAY * torch.sum(
            torch.stack([torch.sum(p ** 2.0) / 2
                         for p in self.network.parameters()])
        )


"""
Helper functions
"""


def scale_gradient(x, scale):
    x.requires_grad_(True)
    x.register_hook(lambda grad: grad * scale)


def cross_entropy_loss(prediction, target):
    return -(target * F.log_softmax(prediction, -1)).sum(-1)
