import time
import ray
import config
import collections
import torch
import torch.nn.functional as F
import numpy as np
from Models.MCTS_Util import split_batch, sample_set_to_idx, create_target_and_mask

Prediction = collections.namedtuple(
    'Prediction',
    'value value_logits reward reward_logits policy_logits comp final_comp champ')

LossOutput = collections.namedtuple(
    'LossOutput',
    'value_loss reward_loss policy_loss tier_loss final_tier_loss champ_loss '
    'value reward policy target_value target_reward l2_loss')


class Trainer(object):
    def __init__(self, global_agent, summary_writer):
        self.network = global_agent
        self.init_learning_rate = config.INIT_LEARNING_RATE
        self.decay_steps = config.WEIGHT_DECAY
        self.alpha = config.LR_DECAY_FUNCTION
        self.optimizer = self.create_optimizer()
        self.summary_writer = summary_writer
        self.model_ckpt_time = time.time_ns()
        self.loss_ckpt_time = time.time_ns()

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

        # observation, action_history, value_mask, reward_mask, policy_mask, target_value, target_reward, target_policy, \
        #     sample_set, tier_set, final_tier_set, champion_set, position = ray.get(batch).tolist()
        observation, action_history, value_mask, reward_mask, policy_mask, target_value, target_reward, target_policy, \
            sample_set, tier_set, final_tier_set, champion_set, position = ray.get(batch)

        # disabling this for the moment while I get the rest working, will add back later.
        self.summary_writer.add_scalar('episode_info/average_position', position, train_step)

        self.loss_ckpt_time = time.time_ns()
        self.adjust_lr(train_step)

        predictions = self.compute_forward(observation, action_history)

        sample_set, target_policy = split_batch(sample_set, target_policy)

        self.compute_loss(predictions, target_value, target_reward, target_policy, sample_set,
                          value_mask, reward_mask, policy_mask, tier_set, final_tier_set, champion_set)

        self.backpropagate()

        self.write_summaries(train_step)
        print("TRAINING TOOK {} time".format(time.time_ns() - self.loss_ckpt_time))

    def compute_forward(self, observation, action_history):
        self.network.train()
        grad_scale = 0.5
        output = self.network.initial_inference(observation, training=True)

        predictions = [
            Prediction(
                value=output["value"],
                value_logits=output["value_logits"],
                # The reward logits in initial inference are not from the network,
                # and might not be on the correct device
                reward=output["reward"].to(config.DEVICE),
                reward_logits=output["reward_logits"].to(config.DEVICE),
                policy_logits=output["policy_logits"],
                comp=output["comp"],
                final_comp=output["final_comp"],
                champ=output["champ"]
            )
        ]

        for unroll_step in range(config.UNROLL_STEPS):
            hidden_state = output["hidden_state"]
            output = self.network.recurrent_inference(hidden_state, action_history[:, unroll_step], training=True)

            scale_dict_gradient(hidden_state, grad_scale)

            predictions.append(
                Prediction(
                    value=output["value"],
                    value_logits=output["value_logits"],
                    reward=output["reward"],
                    reward_logits=output["reward_logits"],
                    policy_logits=output["policy_logits"],
                    comp=output["comp"],
                    final_comp=output["final_comp"],
                    champ=output["champ"]
                )
            )

        return predictions

    def compute_loss(self, predictions, target_value, target_reward, target_policy, sample_set,
                     value_mask, reward_mask, policy_mask, tier_set, final_tier_set, champion_set):
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
            tier_loss=[],
            final_tier_loss=[],
            champ_loss=[],
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

            step_target_policy = self.fill_policy(target_policy[tstep], sample_set[tstep])
            policy_loss = self.policy_loss(prediction.policy_logits, step_target_policy)
            self.scale_loss(policy_loss)
            if config.CHAMP_DECIDER:
                policy_loss.register_hook(lambda grad: grad * (1 / len(config.CHAMPION_ACTION_DIM)))

            # TODO: Figure out how to speed up the tier, final_tier, and champion losses
            tier_target = [tier_set[a][tstep] for a in range(config.BATCH_SIZE)]
            tier_target = [list(b) for b in zip(*tier_target)]
            tier_loss = self.supervised_loss(prediction.comp, tier_target)
            self.scale_loss(tier_loss)
            tier_loss.register_hook(lambda grad: grad * (1 / len(config.TEAM_TIERS_VECTOR)))

            final_tier_target = [final_tier_set[a][tstep] for a in range(config.BATCH_SIZE)]
            final_tier_target = [list(b) for b in zip(*final_tier_target)]
            final_tier_loss = self.supervised_loss(prediction.final_comp, final_tier_target)
            self.scale_loss(final_tier_loss)
            final_tier_loss.register_hook(lambda grad: grad * (1 / len(config.TEAM_TIERS_VECTOR)))

            champion_target = [champion_set[a][tstep] for a in range(config.BATCH_SIZE)]
            champion_target = [list(b) for b in zip(*champion_target)]
            champ_loss = self.supervised_loss(prediction.champ, champion_target)
            self.scale_loss(tier_loss)
            champ_loss.register_hook(lambda grad: grad * (1 / len(config.CHAMPION_ACTION_DIM)))

            # print("Losses tier {} final tier {} champion {}".format(tier_loss, final_tier_loss, champ_loss))

            self.outputs.value_loss.append(value_loss)
            self.outputs.reward_loss.append(reward_loss)
            self.outputs.policy_loss.append(policy_loss)

            self.outputs.tier_loss.append(tier_loss)
            self.outputs.final_tier_loss.append(final_tier_loss)
            self.outputs.champ_loss.append(champ_loss)

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
        tier_loss = torch.stack(self.outputs.tier_loss, -1) * policy_mask
        final_tier_loss = torch.stack(self.outputs.final_tier_loss, -1) * policy_mask
        champ_loss = torch.stack(self.outputs.champ_loss, -1) * policy_mask

        self.loss = torch.sum(
            value_loss * config.VALUE_LOSS_SCALING + reward_loss * config.REWARD_LOSS_SCALING +
            policy_loss * config.POLICY_LOSS_SCALING + tier_loss * config.GAME_METRICS_SCALING +
            final_tier_loss * config.GAME_METRICS_SCALING +
            champ_loss * config.GAME_METRICS_SCALING, -1).to(config.DEVICE)
        # self.loss = torch.sum(
        #     value_loss + reward_loss * config.REWARD_LOSS_SCALING + policy_loss * config.POLICY_LOSS_SCALING,
        #     -1).to(config.DEVICE)

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
            'losses/tier_loss', torch.mean(torch.stack(self.outputs.tier_loss)), train_step)
        self.summary_writer.add_scalar(
            'losses/final_tier_loss', torch.mean(torch.stack(self.outputs.final_tier_loss)), train_step)
        self.summary_writer.add_scalar(
            'losses/champ_loss', torch.mean(torch.stack(self.outputs.champ_loss)), train_step)
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

        # TODO: Figure out a way to get rid of this if statement
        if config.CHAMP_DECIDER:
            for i in range(len(config.CHAMPION_ACTION_DIM)):
                self.summary_writer.add_scalar(
                    'episode_info/value_diff_{}'.format(i),
                    torch.max(torch.max(torch.stack([pol[i] for pol in self.outputs.policy]), 1).values -
                              torch.min(torch.stack([pol[i] for pol in self.outputs.policy]), 1).values), train_step)
        else:
            for i in range(len(config.POLICY_HEAD_SIZES)):
                self.summary_writer.add_scalar(
                    'episode_info/value_diff_{}'.format(i),
                    torch.max(torch.max(torch.stack([pol[i] for pol in self.outputs.policy]), 1).values -
                              torch.min(torch.stack([pol[i] for pol in self.outputs.policy]), 1).values), train_step)

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

    # TODO: Maybe turn this off
    # prediction [ [batch_size, action_dim_1], ...]
    # target [ [batch_size, sampled_action_dim_1], ...] (smaller than prediction)
    # sample_set [ [batch_size, sampled_action_dim_1], ...]
    # We need to mask the prediction so that only the sampled actions are used in the loss.
    # We also need to fill the target with zeros where the prediction is masked.
    def fill_policy(self, target, sample_set):
        if not config.CHAMP_DECIDER:
            idx_set = sample_set_to_idx(sample_set)
            target = create_target_and_mask(target, idx_set)
            target = [torch.from_numpy(target_dim).to(config.DEVICE) for target_dim in target]
        else:
            target = [torch.tensor(target_dim).to(config.DEVICE) for target_dim in target]

        return target

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

    def supervised_loss(self, prediction, target):
        loss = 0.0
        for pred_dim, target_dim in zip(prediction, target):
            # print(pred_dim)
            loss += mean_squared_error_loss(pred_dim, torch.tensor(np.asarray(target_dim, dtype=np.int8),
                                                                   dtype=torch.float32).to(config.DEVICE))
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
    
def scale_dict_gradient(x, scale):
    for key in x:
        x[key].requires_grad_(True)
        x[key].register_hook(lambda grad: grad * scale)


def cross_entropy_loss(prediction, target):
    return -(target * F.log_softmax(prediction, -1)).sum(-1)

def mean_squared_error_loss(prediction, target):
    return F.mse_loss(torch.softmax(prediction, -1), target).sum(-1)

