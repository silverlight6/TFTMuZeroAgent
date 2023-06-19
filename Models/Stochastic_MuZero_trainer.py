import config
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Models.MCTS_Util import split_batch, sample_set_to_idx, create_target_and_mask

Prediction = collections.namedtuple(
    'Prediction',
    'value_logits policy_logits reward_logits, afterstate_value_logits afterstate_policy_logits chance_code chance_embedding'
)

class Trainer:
    def __init__(self, network, summary_writer):
        self.network = network
        self.create_optimizer()
        self.summary_writer = summary_writer
        self.init_learning_rate = config.INIT_LEARNING_RATE
        self.decay_steps = config.WEIGHT_DECAY
        self.alpha = config.LR_DECAY_FUNCTION
        
    def create_optimizer(self):
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.INIT_LEARNING_RATE,
                                          weight_decay=config.WEIGHT_DECAY)

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
        torch.autograd.set_detect_anomaly(True)

        predictions = self.compute_forward(observation, action_history)
        
        sample_set, target_policy = split_batch(sample_set, target_policy)  # [unroll_steps, num_dims, [(batch_size, dim) ...] ]
        
        self.compute_loss(predictions, target_value, target_reward, target_policy, sample_set)
        
        self.backpropagate()
        
        self.summary_writer.add_scalar("loss", self.loss, train_step)
        self.summary_writer.flush()
        
    def compute_forward(self, observation, action_history):
        self.network.train()
        grad_scale = 0.5

        initial_observation = observation[:, 0]
        hidden_state = self.network.representation(initial_observation)
        policy_logits, value_logits = self.network.prediction(hidden_state)
        
        predictions = [ 
            Prediction(
                value_logits=value_logits,
                policy_logits=policy_logits,
                reward_logits=None,
                afterstate_value_logits=None,
                afterstate_policy_logits=None,
                chance_code=None,
                chance_embedding=None
            )
         ]
        
        for unroll_step in range(config.UNROLL_STEPS):
            action = action_history[:, unroll_step]
            
            afterstate = self.network.afterstate_dynamics(hidden_state, action)
            afterstate_policy, afterstate_value_logits = self.network.afterstate_prediction(afterstate)
            chance_code, chance_embedding = self.network.encoder(observation[:, unroll_step + 1]) # Encode the next observation

            next_hidden_state, reward_logits = self.network.dynamics(afterstate, chance_code)
            policy_logits, value_logits = self.network.prediction(next_hidden_state)

            hidden_state = next_hidden_state
            
            self.scale_gradient(hidden_state, grad_scale)
            
            predictions.append(
                Prediction(
                    value_logits=value_logits,
                    policy_logits=policy_logits,
                    reward_logits=reward_logits,
                    afterstate_value_logits=afterstate_value_logits,
                    afterstate_policy_logits=afterstate_policy,
                    chance_code=chance_code,
                    chance_embedding=chance_embedding
                )
            )
            
        return predictions
            
    # gradient_scale = 1.0 / config.UNROLL_STEPS if tstep > 0 else 1.0
    def compute_loss(self, predictions, target_value, target_reward, target_policy, sample_set):
        
        target_value = self.encode_target(target_value, self.network.value_encoder).to(config.DEVICE)
        target_reward = self.encode_target(target_reward, self.network.reward_encoder).to(config.DEVICE)
        
        self.loss = 0.0

        for tstep, prediction in enumerate(predictions):

            step_pred_value, step_target_value = prediction.value_logits, target_value[:, tstep]
            value_loss = self.value_or_reward_loss(step_pred_value, step_target_value)
            self.add_and_scale_loss(value_loss)

            step_pred_policy, step_target_policy = self.mask_and_fill_policy(prediction.policy_logits, target_policy[tstep], sample_set[tstep])
            policy_loss = self.policy_loss(step_pred_policy, step_target_policy)
            self.add_and_scale_loss(policy_loss)
            
            if tstep > 0:
                step_pred_reward, step_target_reward = prediction.reward_logits, target_reward[:, tstep]
                reward_loss = self.value_or_reward_loss(step_pred_reward, step_target_reward)
                self.add_and_scale_loss(reward_loss)
                
                step_afterstate_pred_value, step_afterstate_target_value = prediction.afterstate_value_logits, target_value[:, tstep]
                afterstate_value_loss = self.value_or_reward_loss(step_afterstate_pred_value, step_afterstate_target_value)
                self.add_and_scale_loss(afterstate_value_loss)
                
                step_afterstate_pred_policy, step_chance_code = prediction.afterstate_policy_logits, prediction.chance_code
                afterstate_policy_loss = self.afterstate_distribution_loss(step_afterstate_pred_policy, step_chance_code)
                self.add_and_scale_loss(afterstate_policy_loss)
                
                step_encoder_pred, step_chance_code = prediction.chance_embedding, prediction.chance_code
                encoder_loss = self.vae_commitment_cost(step_encoder_pred, step_chance_code)
                self.add_and_scale_loss(encoder_loss)
                
        self.loss += self.l2_regularization()
        
        self.loss = self.loss.mean()
    
    def backpropagate(self):
        print(self.loss)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
    
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
            (-1, target.shape[-1] , int(encoder.num_steps))
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
        

    def scale_gradient(self, x, scale):
        x.register_hook(lambda grad: grad * scale)
        
    def add_and_scale_loss(self, loss):
        self.scale_gradient(loss, 1.0 / config.UNROLL_STEPS)
        self.loss += loss
        
    
    # Loss for each type of prediction
    def value_or_reward_loss(self, prediction, target):
        return self.cross_entropy(prediction, target)
    
    def policy_loss(self, prediction, target):
        loss = 0.0
        for pred_dim, target_dim in zip(prediction, target):
            loss += self.cross_entropy(pred_dim, target_dim)
        return loss
    
    def afterstate_distribution_loss(self, prediction, target):
        return self.kl_divergence(prediction, target)
    
    def vae_commitment_cost(self, prediction, target):
        return self.mse(prediction, target)
    
    def l2_regularization(self):
        return config.WEIGHT_DECAY * torch.sum(
                torch.stack([torch.sum(p ** 2.0) / 2 
                             for p in self.network.parameters()])
        )
    
    # Loss functions
    def cross_entropy(self, prediction, target):
        return -(target * F.log_softmax(prediction, -1)).sum(-1)
    
    # sum of { p * log ( p / q ) }
    def kl_divergence(self, prediction, target):
        # Add a small value to prevent log(0)
        prediction = prediction + 1e-9
        target = target + 1e-9

        return (target * (torch.log(target) - F.log_softmax(prediction, -1))).sum(-1)
    
    def mse(self, prediction, target):
        return ( ( prediction - target ) ** 2 ).mean(-1)