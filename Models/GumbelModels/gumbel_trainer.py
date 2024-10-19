import copy
import time
import config
import collections
import torch
import numpy as np
from typing import Dict, Any
from torch.nn import KLDivLoss

Prediction = collections.namedtuple(
    'Prediction',
    'value value_logits reward reward_logits policy_logits comp final_comp champ')

LossOutput = collections.namedtuple(
    'LossOutput',
    'value_loss reward_loss policy_loss tier_loss final_tier_loss champ_loss '
    'value reward policy target_value target_reward target_policy l2_loss importance_weights')


class Trainer(object):
    def __init__(self, global_agent, summary_writer):
        self.network = global_agent
        self.decay_steps = config.WEIGHT_DECAY
        self.alpha = config.LR_DECAY_FUNCTION
        self.lr_scheduler = None
        self.optimizer = self.create_optimizer()
        self.summary_writer = summary_writer
        self.model_ckpt_time = time.time_ns()
        self.loss_ckpt_time = time.time_ns()
        self.target_model = copy.deepcopy(self.network)
        self.learn_model = self.network
        self.kl_loss = KLDivLoss(reduction='none')
        self.batch_size = config.BATCH_SIZE

    def create_optimizer(self):
        optimizer = torch.optim.Adam(self.network.parameters(), lr=config.INIT_LEARNING_RATE,
                                     weight_decay=config.WEIGHT_DECAY)

        from torch.optim.lr_scheduler import LambdaLR
        max_step = int(5e4)
        # NOTE: the 1, 0.1, 0.01 is the decay rate, not the lr.
        lr_lambda = lambda step: 1 if step < max_step * 0.5 else (0.1 if step < max_step else 0.01)  # noqa
        self.lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        return optimizer

    def train_network(self, batch, train_step):
        observation, action_batch, policy_mask, target_value, target_reward, target_policy, \
            importance_weights, position = batch

        # disabling this for the moment while I get the rest working, will add back later.
        self.summary_writer.add_scalar('episode_info/average_position', position, train_step)

        summaries = self.compute_loss(observation, action_batch, target_value, target_reward, target_policy,
                                      policy_mask, importance_weights)

        self.write_summaries(train_step, summaries)

    def compute_loss(self, observation, action_batch, target_value, target_reward, target_policy,
                     policy_mask, importance_weights):
        self.network.train()

        policy_mask = torch.from_numpy(policy_mask).to(config.DEVICE)
        weights = torch.from_numpy(importance_weights).to(config.DEVICE)

        # I can add the transformed values to tensorboard later if needed
        target_encoded_value = self.encode_target(
            scalar_transform(target_value), self.network.value_encoder).to(config.DEVICE)
        target_encoded_reward = self.encode_target(
            scalar_transform(target_reward), self.network.reward_encoder).to(config.DEVICE)

        output = self.network.initial_inference(observation, training=True)

        original_value = output["value"]

        # Use this if need to readjust priorities to reuse sample data.
        # value_priority = L1Loss(reduction='none')(original_value, target_value[:, 0])
        # value_priority = value_priority.data.cpu().numpy() + 1e-6

        policy_logits = output["policy_logits"]
        policy_loss = self.kl_loss(torch.log(torch.softmax(policy_logits, dim=1)),
                                   torch.from_numpy(target_policy[:, 0]).to(config.DEVICE).detach().float())
        policy_loss = policy_loss.mean(dim=-1) * policy_mask[:, 0]
        # Output the entropy for experimental observation.

        policy_entropy = -(torch.softmax(policy_logits, dim=-1) * torch.log_softmax(policy_logits, dim=-1)).sum(-1)

        value_loss = cross_entropy_loss(output["value_logits"], target_encoded_value[:, 0])

        reward_loss = torch.zeros(self.batch_size, device=config.DEVICE)

        for step_k in range(5):  # Number of unroll steps
            # unroll with the dynamics function: predict the next ``latent_state``, ``reward``,
            # given current ``latent_state`` and ``action``.
            # And then predict policy_logits and value with the prediction function.
            output = self.learn_model.recurrent_inference(output["hidden_state"], action_batch[:, step_k])

            # NOTE: the target policy, target_value_categorical, target_reward_categorical is calculated in
            # game buffer now.
            # ==============================================================
            # calculate policy loss for the next ``num_unroll_steps`` unroll steps.
            # NOTE: the +=.
            # ==============================================================
            policy_loss += self.kl_loss(torch.log(torch.softmax(output["policy_logits"], dim=1)),
                                        torch.from_numpy(target_policy[:, step_k + 1]).to(
                                            config.DEVICE).detach().float()).mean(dim=-1) * policy_mask[:, step_k + 1]
            value_loss += cross_entropy_loss(output["value_logits"], target_encoded_value[:, step_k + 1])
            reward_loss += cross_entropy_loss(output["reward_logits"], target_encoded_reward[:, step_k])

            policy_entropy += -(torch.softmax(output["policy_logits"], dim=-1) *
                                torch.log_softmax(output["policy_logits"], dim=-1)).sum(-1)

        # ==============================================================
        # the core learn model update step.
        # ==============================================================
        # weighted loss with masks (some invalid states which are out of trajectory.)
        # policy_loss_weight --> 1, value_loss_weight --> 0.25, reward_loss-weight --> 1
        loss = (
                config.POLICY_LOSS_SCALING * policy_loss + config.VALUE_LOSS_SCALING * value_loss +
                config.REWARD_LOSS_SCALING * reward_loss
        )

        weighted_total_loss = (weights * loss).mean()

        gradient_scale = 1 / config.UNROLL_STEPS
        weighted_total_loss.register_hook(lambda grad: grad * gradient_scale)
        self.optimizer.zero_grad()
        weighted_total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.learn_model.parameters(), config.MAX_GRAD_NORM)
        self.optimizer.step()
        self.lr_scheduler.step()

        # ==============================================================
        # the core target model update step.
        # ==============================================================
        return {
            'cur_lr': self.optimizer.param_groups[0]['lr'],
            'weighted_total_loss': weighted_total_loss.item(),
            'total_loss': loss.mean().item(),
            'policy_loss': policy_loss.mean().item(),
            'reward_loss': reward_loss.mean().item(),
            'value_loss': value_loss.mean().item(),
            'policy_entropy': policy_entropy.mean().item(),

            # ==============================================================
            # priority related
            # ==============================================================
            # 'value_priority_orig': value_priority,
            # 'value_priority': value_priority.mean().item(),
            # 'target_reward': target_reward.detach().cpu().numpy().mean().item(),
            # 'target_value': target_value.detach().cpu().numpy().mean().item(),
            'predicted_value': original_value.mean().item(),
            'predicted_policy': torch.softmax(policy_logits, dim=-1),
            'target_policy': target_policy,
            'policy': policy_logits,
            # 'total_grad_norm_before_clip': total_grad_norm_before_clip.item()
        }

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

    def _state_dict_learn(self) -> Dict[str, Any]:
        """
        Overview:
            Return the state_dict of learn mode, usually including model, target_model and optimizer.
        Returns:
            - state_dict (:obj:`Dict[str, Any]`): The dict of current policy learn state, for saving and restoring.
        """
        return {
            'model': self.learn_model.state_dict(),
            'target_model': self.target_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        """
        Overview:
            Load the state_dict variable into policy learn mode.
        Arguments:
            - state_dict (:obj:`Dict[str, Any]`): The dict of policy learn state saved before.
        """
        self.learn_model.load_state_dict(state_dict['model'])
        self.target_model.load_state_dict(state_dict['target_model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def write_summaries(self, train_step, summaries):
        self.summary_writer.add_scalar('losses/weighted_total', summaries["weighted_total_loss"], train_step)
        self.summary_writer.add_scalar('losses/total', summaries["total_loss"], train_step)

        self.summary_writer.add_scalar(
            'losses/value', summaries["value_loss"], train_step)
        self.summary_writer.add_scalar(
            'losses/reward', summaries["reward_loss"], train_step)
        self.summary_writer.add_scalar(
            'losses/policy', summaries["policy_loss"], train_step)
        self.summary_writer.add_scalar(
            'losses/policy_entropy', summaries["policy_entropy"], train_step)

        self.summary_writer.add_scalar(
            'episode_info/value_diff',
            torch.max(torch.max(summaries["policy"], 1).values -
                      torch.min(summaries["policy"], 1).values), train_step)
        self.summary_writer.add_image('policy_scale', torch.sum(torch.reshape(torch.tensor(
            summaries["target_policy"]).to(config.DEVICE), (-1, 55, 38)), dim=0)[None, :, :], global_step=train_step)
        self.summary_writer.add_image('policy_preference', torch.mul(torch.mean(torch.reshape(
            summaries["predicted_policy"], (-1, 55, 38)), dim=0), 100)[None, :, :], global_step=train_step)

        self.summary_writer.flush()

def scalar_transform(x: np.array, epsilon: float = 0.001, delta: float = 1.) -> np.array:
    """
    Overview:
        Transform the original value to the scaled value, i.e. the h(.) function
        in paper https://arxiv.org/pdf/1805.11593.pdf.
    Reference:
        - MuZero: Appendix F: Network Architecture
        - https://arxiv.org/pdf/1805.11593.pdf (Page-11) Appendix A : Proposition A.2
    """
    # h(.) function
    if delta == 1:  # for speed up
        output = np.sign(x) * (np.sqrt(np.abs(x) + 1) - 1) + epsilon * x
    else:
        # delta != 1
        output = np.sign(x) * (np.sqrt(np.abs(x / delta) + 1) - 1) + epsilon * x / delta
    return output


"""
Helper functions
"""


def cross_entropy_loss(prediction, target):
    return -(torch.log_softmax(prediction, dim=1) * target).sum(1)
