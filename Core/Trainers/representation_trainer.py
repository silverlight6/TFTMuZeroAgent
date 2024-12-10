import time
import config
import collections
import torch
import torch.nn.functional as F
import numpy as np
from Simulator.batch_generator import BatchGenerator
from Simulator.tft_config import TFTConfig

Prediction = collections.namedtuple(
    'Prediction',
    'comp champ')
# Prediction = collections.namedtuple(
#     'Prediction',
#     'comp champ shop, item scalar')

# LossOutput = collections.namedtuple(
#     'LossOutput',
#     'comp_loss  champ_loss shop_loss item_loss scalar_loss l2_loss, loss')

LossOutput = collections.namedtuple(
    'LossOutput',
    'comp_loss  champ_loss loss')


class RepresentationTrainer(object):
    def __init__(self, global_agent, summary_writer):
        self.network = global_agent
        self.init_learning_rate = config.INIT_LEARNING_RATE
        self.decay_steps = config.DECAY_STEPS
        self.alpha = config.LR_DECAY_FUNCTION
        self.summary_writer = summary_writer
        self.model_ckpt_time = time.time_ns()
        self.loss_ckpt_time = time.time_ns()

        tftConfig = TFTConfig()
        from Simulator.observation.token.basic_observation import ObservationToken
        tftConfig.observation_class = ObservationToken

        self.batch_generator = BatchGenerator(tftConfig)
        # self.harmony_comp = torch.nn.Parameter(-torch.log(torch.tensor(1.0)))
        # self.harmony_champ = torch.nn.Parameter(-torch.log(torch.tensor(1.0)))
        # self.harmony_shop = torch.nn.Parameter(-torch.log(torch.tensor(1.0)))
        # self.harmony_item = torch.nn.Parameter(-torch.log(torch.tensor(1.0)))
        # self.harmony_scalar = torch.nn.Parameter(-torch.log(torch.tensor(1.0)))
        self.optimizer = self.create_optimizer()
        self.softmax = torch.nn.Softmax()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.network.train()

    def create_optimizer(self):
        # parameters = list(self.network.parameters()) + [
        #     self.harmony_comp,
        #     self.harmony_champ,
        #     self.harmony_shop,
        #     self.harmony_item,
        #     self.harmony_scalar
        # ]
        # parameters = list(self.network.parameters())

        # optimizer = torch.optim.Adam(parameters, lr=config.INIT_LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=config.INIT_LEARNING_RATE)
        return optimizer

    def decayed_learning_rate(self, step):
        # Calculate the decay factor as 10^n where n is the number of thresholds crossed
        decay_factor = 10 ** (step / 300)  # 10x at 1000, 100x at 10000, etc.

        # Apply decay to the initial learning rate
        decayed_learning_rate = self.init_learning_rate / decay_factor

        # Return the decayed learning rate
        return decayed_learning_rate

    # Same as muzero-general
    def adjust_lr(self, train_step):
        lr = self.decayed_learning_rate(train_step)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def train_network(self, train_step):

        observation, labels = self.batch_generator.generate_batch(batch_size=config.BATCH_SIZE)

        # self.adjust_lr(train_step)

        predictions = self.compute_forward(observation)

        self.compute_loss(predictions, labels)

        self.write_summaries(train_step)

    def compute_forward(self, observation):

        output = self.network(observation)

        predictions = Prediction(
            comp=output["comp"],
            champ=output["champ"],
            # shop=output["shop"],
            # item=output["item"],
            # scalar=output["scalar"]
        )

        return predictions

    def compute_loss(self, predictions, labels):
        self.outputs = LossOutput(
            comp_loss=[],
            champ_loss=[],
            loss=[]
        )

        # TODO: Figure out how to speed up the tier, final_tier, and champion losses
        comp_loss = 0
        comp_target = [label[0] for label in labels]
        comp_target = [[np.argmax(label[i]) for label in comp_target]
                       for i in range(len(config.TEAM_TIERS_VECTOR))]
        for output, target in zip(predictions.comp, comp_target):
            comp_loss += self.criterion(output, torch.tensor(target, dtype=torch.long).to(config.DEVICE))
        # comp_loss = self.supervised_loss(predictions.comp, comp_target)

        champ_loss = 0
        champion_target = [label[1] for label in labels]
        champion_target = [[np.argmax(label[i]) for label in champion_target]
                           for i in range(len(config.CHAMPION_LIST_DIM))]
        for output, target in zip(predictions.champ, champion_target):
            champ_loss += self.criterion(output, torch.tensor(target, dtype=torch.long).to(config.DEVICE))
        # champ_loss = self.supervised_loss(predictions.champ, champion_target)

        # shop_target = [label[2] for label in labels]
        # shop_target = [list(b) for b in zip(*shop_target)]
        # shop_loss = self.supervised_loss(predictions.shop, shop_target)

        # item_target = [label[3] for label in labels]
        # item_target = [list(b) for b in zip(*item_target)]
        # item_loss = self.supervised_loss(predictions.item, item_target)

        # scalar_target = [label[4] for label in labels]
        # scalar_target = [list(b) for b in zip(*scalar_target)]
        # scalar_loss = self.supervised_loss(predictions.scalar, scalar_target)

        # l2_loss = self.l2_regularization()
        # self.outputs.l2_loss.append(l2_loss)

        self.outputs.comp_loss.append(comp_loss)
        self.outputs.champ_loss.append(champ_loss)
        # self.outputs.shop_loss.append(shop_loss)
        # self.outputs.item_loss.append(item_loss)
        # self.outputs.scalar_loss.append(scalar_loss)

        tier_loss = torch.stack(self.outputs.comp_loss, -1)
        champ_loss = torch.stack(self.outputs.champ_loss, -1)
        # shop_loss = torch.stack(self.outputs.shop_loss, -1)
        # item_loss = torch.stack(self.outputs.item_loss, -1)
        # scalar_loss = torch.stack(self.outputs.scalar_loss, -1)

        # loss = (
        #         (tier_loss.mean() / torch.exp(self.harmony_comp))
        #         + (champ_loss.mean() / torch.exp(self.harmony_champ))
        #         + (shop_loss.mean() / torch.exp(self.harmony_shop))
        #         + (item_loss.mean() / torch.exp(self.harmony_item))
        #         + (scalar_loss.mean() / torch.exp(self.harmony_scalar))
        # )
        # weighted_total_loss = loss.mean()
        # weighted_total_loss += (
        #         torch.log(torch.exp(self.harmony_comp) + 1) +
        #         torch.log(torch.exp(self.harmony_champ) + 1) +
        #         torch.log(torch.exp(self.harmony_shop) + 1) +
        #         torch.log(torch.exp(self.harmony_item) + 1) +
        #         torch.log(torch.exp(self.harmony_scalar) + 1)
        # )
        # weighted_total_loss = tier_loss.mean() + champ_loss.mean() + shop_loss.mean() + item_loss.mean() + scalar_loss.mean()
        loss = tier_loss + champ_loss
        # weighted_total_loss += l2_loss

        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.outputs.loss.append(loss)

    def write_summaries(self, train_step):
        # print(f"tier_loss is {self.outputs.comp_loss}, champ_loss is {self.outputs.champ_loss}, "
        #       f"shop_loss is {self.outputs.shop_loss}, item_loss is {self.outputs.item_loss}, "
        #       f"scalar_loss is {self.outputs.scalar_loss}, learning rate {self.optimizer.param_groups[0]['lr']}, "
        #       f"total loss is {self.outputs.loss} at time step {train_step}")
        print(f"tier_loss is {self.outputs.comp_loss}, champ_loss is {self.outputs.champ_loss}, "
              f"learning rate {self.optimizer.param_groups[0]['lr']}, "
              f"total loss is {self.outputs.loss} at time step {train_step}")
        self.summary_writer.add_scalar(
            'losses/total', torch.mean(torch.stack(self.outputs.loss)), train_step)
        self.summary_writer.add_scalar(
            'losses/tier_loss', torch.mean(torch.stack(self.outputs.comp_loss)), train_step)
        self.summary_writer.add_scalar(
            'losses/champ_loss', torch.mean(torch.stack(self.outputs.champ_loss)), train_step)
        # self.summary_writer.add_scalar(
        #     'losses/shop_loss', torch.mean(torch.stack(self.outputs.shop_loss)), train_step)
        # self.summary_writer.add_scalar(
        #     'losses/item_loss', torch.mean(torch.stack(self.outputs.item_loss)), train_step)
        # self.summary_writer.add_scalar(
        #     'losses/scalar_loss', torch.mean(torch.stack(self.outputs.scalar_loss)), train_step)
        # self.summary_writer.add_scalar(
        #     'losses/l2', torch.mean(torch.stack(self.outputs.l2_loss)), train_step)
        # self.summary_writer.add_scalar(
        #     'weights/harmony_comp', self.harmony_comp.item(), train_step)
        # self.summary_writer.add_scalar(
        #     'weights/harmony_comp_exp_recip', (1 / torch.exp(self.harmony_comp)).item(), train_step)
        # self.summary_writer.add_scalar(
        #     'weights/harmony_champ', self.harmony_champ.item(), train_step)
        # self.summary_writer.add_scalar(
        #     'weights/harmony_champ_exp_recip', (1 / torch.exp(self.harmony_champ)).item(), train_step)
        # self.summary_writer.add_scalar(
        #     'weights/harmony_shop', self.harmony_shop.item(), train_step)
        # self.summary_writer.add_scalar(
        #     'weights/harmony_shop_exp_recip', (1 / torch.exp(self.harmony_shop)).item(), train_step)
        # self.summary_writer.add_scalar(
        #     'weights/harmony_item', self.harmony_item.item(), train_step)
        # self.summary_writer.add_scalar(
        #     'weights/harmony_item_exp_recip', (1 / torch.exp(self.harmony_item)).item(), train_step)
        # self.summary_writer.add_scalar(
        #     'weights/harmony_scalar', self.harmony_scalar.item(), train_step)
        # self.summary_writer.add_scalar(
        #     'weights/harmony_scalar_exp_recip', (1 / torch.exp(self.harmony_scalar)).item(), train_step)
        self.summary_writer.add_scalar(
            'training_values/learning_rate', self.optimizer.param_groups[0]['lr'], train_step)

        # self.summary_writer.flush()

        if train_step % config.CHECKPOINT_STEPS == 0:
            self.network.tft_save_model(train_step, self.optimizer)

    def supervised_loss(self, prediction, target):
        loss = 0.0
        for pred_dim, target_dim in zip(prediction, target):
            tensor_target = torch.tensor(np.asarray(target_dim, dtype=np.int8),
                                                      dtype=torch.float32).to(config.DEVICE)
            loss += self.criterion(pred_dim, tensor_target)
        return loss

    def l2_regularization(self):
        return config.WEIGHT_DECAY * torch.sum(
            torch.stack([torch.sum(p ** 2.0) / 2
                         for p in self.network.parameters()])
        )

def mean_squared_error_loss(prediction, target):
    return F.mse_loss(torch.softmax(prediction, -1), target).sum(-1)

