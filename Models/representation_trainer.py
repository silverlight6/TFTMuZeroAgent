import time
import config
import collections
import torch
import torch.nn.functional as F
import numpy as np
from Simulator.batch_generator import BatchGenerator

# Prediction = collections.namedtuple(
#     'Prediction',
#     'comp champ shop, item scalar')
Prediction = collections.namedtuple(
    'Prediction',
    'comp')

LossOutput = collections.namedtuple(
    'LossOutput',
    'comp_loss  champ_loss shop_loss item_loss scalar_loss l2_loss')


class RepresentationTrainer(object):
    def __init__(self, global_agent, summary_writer):
        self.network = global_agent
        self.init_learning_rate = config.INIT_LEARNING_RATE
        self.decay_steps = config.DECAY_STEPS
        self.alpha = config.LR_DECAY_FUNCTION
        self.optimizer = self.create_optimizer()
        self.summary_writer = summary_writer
        self.model_ckpt_time = time.time_ns()
        self.loss_ckpt_time = time.time_ns()
        self.batch_generator = BatchGenerator()

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

    def train_network(self, train_step):

        observation, labels = self.batch_generator.generate_batch(batch_size=config.BATCH_SIZE)

        self.adjust_lr(train_step)

        predictions = self.compute_forward(observation)

        self.compute_loss(predictions, labels)

        self.backpropagate()

        self.write_summaries(train_step)

    def compute_forward(self, observation):
        self.network.train()
        output = self.network.forward(observation)

        # predictions = Prediction(
        #     comp=output["comp"],
        #     champ=output["champ"],
        #     shop=output["shop"],
        #     item=output["item"],
        #     scalar=output["scalar"]
        # )
        predictions = Prediction(
            comp=output["comp"]
        )

        return predictions

    def compute_loss(self, predictions, labels):
        self.outputs = LossOutput(
            comp_loss=[],
            champ_loss=[],
            shop_loss=[],
            item_loss=[],
            scalar_loss=[],
            l2_loss=[],
        )

        # TODO: Figure out how to speed up the tier, final_tier, and champion losses
        comp_target = [label[0] for label in labels]
        comp_target = [list(b) for b in zip(*comp_target)]
        comp_loss = self.supervised_loss(predictions.comp, comp_target)

        # champion_target = [label[1] for label in labels]
        # champion_target = [list(b) for b in zip(*champion_target)]
        # champ_loss = self.supervised_loss(predictions.champ, champion_target)
        #
        # shop_target = [label[2] for label in labels]
        # shop_target = [list(b) for b in zip(*shop_target)]
        # shop_loss = self.supervised_loss(predictions.shop, shop_target)
        #
        # item_target = [label[3] for label in labels]
        # item_target = [list(b) for b in zip(*item_target)]
        # item_loss = self.supervised_loss(predictions.item, item_target)
        #
        # scalar_target = [label[4] for label in labels]
        # scalar_target = [list(b) for b in zip(*scalar_target)]
        # scalar_loss = self.supervised_loss(predictions.scalar, scalar_target)

        # print("Losses tier {} final tier {} champion {}".format(tier_loss, final_tier_loss, champ_loss))

        l2_loss = self.l2_regularization()
        self.outputs.l2_loss.append(l2_loss)

        self.outputs.comp_loss.append(comp_loss)
        # self.outputs.champ_loss.append(champ_loss)
        # self.outputs.shop_loss.append(shop_loss)
        # self.outputs.item_loss.append(item_loss)
        # self.outputs.scalar_loss.append(scalar_loss)

        tier_loss = torch.stack(self.outputs.comp_loss, -1)
        # champ_loss = torch.stack(self.outputs.champ_loss, -1)
        # shop_loss = torch.stack(self.outputs.shop_loss, -1)
        # item_loss = torch.stack(self.outputs.item_loss, -1)
        # scalar_loss = torch.stack(self.outputs.scalar_loss, -1)

        # self.loss = torch.sum(tier_loss + champ_loss + shop_loss + item_loss + scalar_loss, -1).to(config.DEVICE)
        self.loss = torch.sum(tier_loss, -1).to(config.DEVICE)

        self.loss = self.loss.mean()
        self.loss += l2_loss

    def backpropagate(self):
        self.loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def write_summaries(self, train_step):
        # print(f"tier_loss is {self.outputs.comp_loss}, champ_loss is {self.outputs.champ_loss}, "
        #       f"shop_loss is {self.outputs.shop_loss}, item_loss is {self.outputs.item_loss}, "
        #       f"scalar_loss is {self.outputs.scalar_loss}, learning rate {self.optimizer.param_groups[0]['lr']}, "
        #       f"total loss is {self.loss} at time step {train_step}")
        print(f"tier_loss is {self.outputs.comp_loss}, learning rate {self.optimizer.param_groups[0]['lr']}, "
              f"total loss is {self.loss} at time step {train_step}")
        self.summary_writer.add_scalar('losses/total', self.loss, train_step)

        self.summary_writer.add_scalar(
            'losses/tier_loss', torch.mean(torch.stack(self.outputs.comp_loss)), train_step)
        # self.summary_writer.add_scalar(
        #     'losses/champ_loss', torch.mean(torch.stack(self.outputs.champ_loss)), train_step)
        # self.summary_writer.add_scalar(
        #     'losses/shop_loss', torch.mean(torch.stack(self.outputs.shop_loss)), train_step)
        # self.summary_writer.add_scalar(
        #     'losses/item_loss', torch.mean(torch.stack(self.outputs.item_loss)), train_step)
        # self.summary_writer.add_scalar(
        #     'losses/scalar_loss', torch.mean(torch.stack(self.outputs.scalar_loss)), train_step)
        self.summary_writer.add_scalar(
            'losses/l2', torch.mean(torch.stack(self.outputs.l2_loss)), train_step)

        self.summary_writer.flush()

        if train_step % 10 == 0:
            self.network.tft_save_model(train_step)

    def supervised_loss(self, prediction, target):
        loss = 0.0
        for pred_dim, target_dim in zip(prediction, target):
            loss += torch.nn.functional.cross_entropy(pred_dim, torch.tensor(np.asarray(target_dim, dtype=np.int8),
                                                      dtype=torch.float32).to(config.DEVICE))
        return loss

    def l2_regularization(self):
        return config.WEIGHT_DECAY * torch.sum(
            torch.stack([torch.sum(p ** 2.0) / 2
                         for p in self.network.parameters()])
        )

def mean_squared_error_loss(prediction, target):
    return F.mse_loss(torch.softmax(prediction, -1), target).sum(-1)

