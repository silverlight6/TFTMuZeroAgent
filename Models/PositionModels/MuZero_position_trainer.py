import copy
import time
import config
import collections
import torch
import torch.nn.functional as F
import numpy as np
import copy

Prediction = collections.namedtuple(
    'Prediction',
    'value value_logits policy_logits')

LossOutput = collections.namedtuple(
    'LossOutput',
    'value_loss policy_loss value policy policy_entropy target_value target_policy l2_loss importance_weights')


class Trainer(object):
    def __init__(self, global_agent, summary_writer, optimizer_dict=None):
        self.network = global_agent
        self.init_learning_rate = config.INIT_LEARNING_RATE
        self.decay_steps = config.WEIGHT_DECAY
        self.alpha = config.LR_DECAY_FUNCTION
        self.optimizer = self.create_optimizer(optimizer_dict)
        self.summary_writer = summary_writer
        self.model_ckpt_time = time.time_ns()
        self.loss_ckpt_time = time.time_ns()

    def create_optimizer(self, optimizer_dict=None):
        optimizer = torch.optim.Adam(self.network.parameters(), lr=config.INIT_LEARNING_RATE,
                                     weight_decay=config.WEIGHT_DECAY)
        if optimizer_dict is not None:
            optimizer.load_state_dict(optimizer_dict)
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

        observation, action_history, value_mask, policy_mask, target_value, target_policy = batch

        self.adjust_lr(train_step)

        predictions = self.compute_forward(observation, action_history)

        self.compute_loss(predictions, target_value, target_policy,  value_mask, policy_mask)

        self.backpropagate()

        self.write_summaries(train_step)

    def compute_forward(self, observation, action_history):
        self.network.train()
        grad_scale = 0.5
        output = self.network.initial_inference(observation, training=True)

        predictions = [
            Prediction(
                value=output["value"],
                value_logits=output["value_logits"],
                policy_logits=output["policy_logits"],
            )
        ]

        for unroll_step in range(config.UNROLL_STEPS):
            hidden_state = output["hidden_state"]
            output = self.network.recurrent_inference(hidden_state, action_history[:, unroll_step])

            scale_dict_gradient(hidden_state, grad_scale)

            predictions.append(
                Prediction(
                    value=output["value"],
                    value_logits=output["value_logits"],
                    policy_logits=output["policy_logits"],
                )
            )

        return predictions

    def compute_loss(self, predictions, target_value, target_policy, value_mask, policy_mask):
        value_mask = torch.from_numpy(value_mask).to(config.DEVICE)
        policy_mask = torch.from_numpy(policy_mask).to(config.DEVICE)
        target_policy = torch.from_numpy(target_policy).to(config.DEVICE)

        target_value = self.encode_target(
            target_value, self.network.value_encoder).to(config.DEVICE)

        self.outputs = LossOutput(
            value_loss=[],
            policy_loss=[],
            value=[],
            policy=[],
            policy_entropy=[],
            target_value=[],
            target_policy=[],
            l2_loss=[],
            importance_weights=[]
        )

        for tstep, prediction in enumerate(predictions):
            step_value, step_target_value = prediction.value_logits, target_value[:, tstep]
            value_loss = cross_entropy_loss(step_value, step_target_value)

            policy_loss = cross_entropy_loss(prediction.policy_logits, target_policy[:, tstep])
            prob = torch.softmax(prediction.policy_logits, dim=-1)
            policy_entropy = -(prob * prob.log()).sum(-1)
            policy_entropy_loss = policy_entropy * -1

            self.outputs.value_loss.append(value_loss)
            self.outputs.policy_loss.append(policy_loss)
            self.outputs.policy_entropy.append(policy_entropy_loss)

            self.outputs.value.append(prediction.value)
            self.outputs.policy.append(prediction.policy_logits)

            self.outputs.target_value.append(self.decode_target(step_target_value, self.network.value_encoder))

        l2_loss = self.l2_regularization()
        self.outputs.l2_loss.append(l2_loss)

        value_loss = torch.stack(self.outputs.value_loss, -1) * value_mask
        policy_loss = torch.stack(self.outputs.policy_loss, -1) * policy_mask
        entropy_loss = torch.stack(self.outputs.policy_entropy, -1) * policy_mask

        self.loss = torch.sum(
            value_loss * config.VALUE_LOSS_SCALING + policy_loss * config.POLICY_LOSS_SCALING + entropy_loss * 0.1,
            -1).to(config.DEVICE)

        self.loss = self.loss.mean()
        self.scale_loss(self.loss)
        # self.loss += l2_loss


    def backpropagate(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def write_summaries(self, train_step):
        self.summary_writer.add_scalar('losses/total', self.loss, train_step)

        self.summary_writer.add_scalar(
            'losses/value', torch.mean(torch.stack(self.outputs.value_loss)), train_step)
        self.summary_writer.add_scalar(
            'losses/policy', torch.mean(torch.stack(self.outputs.policy_loss)), train_step)
        self.summary_writer.add_scalar(
            'losses/l2', torch.mean(torch.stack(self.outputs.l2_loss)), train_step)

        self.summary_writer.add_scalar(
            'prediction/value', torch.mean(torch.stack(self.outputs.value)), train_step)
        self.summary_writer.add_scalar(
            'target/value', torch.mean(torch.stack(self.outputs.target_value)), train_step)
        self.summary_writer.add_scalar(
            'episode_max/value', torch.max(torch.stack(self.outputs.target_value)), train_step)
        self.summary_writer.add_scalar(
            'episode_info/policy_entropy', torch.mean(torch.stack(self.outputs.policy_entropy)), train_step)

        self.summary_writer.add_scalar(
            'episode_info/value_diff',
            torch.max(torch.max(torch.stack(self.outputs.policy), 1).values -
                      torch.min(torch.stack(self.outputs.policy), 1).values), train_step)

        # print(f"loss {self.loss}, target_value {torch.mean(torch.stack(self.outputs.target_value))}")

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

    def scale_loss(self, loss):
        scale_gradient(loss, 1.0 / config.UNROLL_STEPS)

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
    
def scale_dict_gradient(x, scale):
    x.requires_grad_(True)
    x.register_hook(lambda grad: grad * scale)


def cross_entropy_loss(prediction, target):
    return -(torch.log_softmax(prediction, dim=1) * target).sum(-1)
