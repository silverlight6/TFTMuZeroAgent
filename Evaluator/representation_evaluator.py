import config
import collections
import torch
import numpy as np
import time
from Simulator.batch_generator import BatchGenerator
from sklearn.metrics import confusion_matrix
from Evaluator.eval_visualizers import show_confustion_matrix


# Prediction = collections.namedtuple(
#     'Prediction',
#     'comp champ shop, item scalar')
Prediction = collections.namedtuple(
    'Prediction',
    'comp')

class RepresentationEvaluator:
    def __init__(self, global_agent):
        self.network = global_agent
        self.batch_generator = BatchGenerator()
        self.softmax = torch.nn.Softmax(dim=-1)

    def evaluate(self):
        observation, labels = self.batch_generator.generate_batch(batch_size=config.BATCH_SIZE)
        predictions = self.compute_forward(observation)

        self.create_graphs(predictions, labels)

    def compute_forward(self, observation):
        self.network.train()
        output = self.network.forward(observation)
        comp_output_list = torch.tensor([])
        for t in output["comp"]:
            for comp in t:
                max_index = torch.argmax(self.softmax(comp))

                one_hot = torch.zeros_like(comp)  # Create a tensor of zeros
                one_hot[max_index] = 1  # Set the element at the max index to 1

                # Append the one-hot encoded tensor to the results list
                comp_output_list = torch.cat((comp_output_list, one_hot.detach().cpu()))

        predictions = Prediction(
            comp=comp_output_list.numpy(),
            # champ=torch.argmax(torch.cat([t.flatten() for t in output["champ"]]), -1).to(dtype=torch.int8).cpu(),
            # shop=torch.argmax(torch.cat([t.flatten() for t in output["shop"]]), -1).to(dtype=torch.int8).cpu(),
            # item=torch.argmax(torch.cat([t.flatten() for t in output["item"]]), -1).to(dtype=torch.int8).cpu(),
            # scalar=torch.argmax(torch.cat([t.flatten() for t in output["scalar"]]), -1).to(dtype=torch.int8).cpu()
        )
        return predictions

    def create_graphs(self, pred, labels):
        # TODO: Figure out how to speed up the tier, final_tier, and champion losses
        pred_comp = pred.comp
        # pred_comp = (pred_comp > 0.5).astype(int)
        tier_target = [label[0] for label in labels]
        tier_target = np.concatenate([list(b) for b in zip(*tier_target)], axis=1).flatten()
        comp_cm = confusion_matrix(pred_comp, tier_target)
        show_confustion_matrix(comp_cm)

        # champion_target = [label[1] for label in labels]
        # champion_target = np.concatenate([list(b) for b in zip(*champion_target)], axis=1)
        # champ_cm = confusion_matrix(pred.champ, champion_target)
        # show_confustion_matrix(champ_cm)
        #
        # shop_target = [label[2] for label in labels]
        # shop_target = np.concatenate([list(b) for b in zip(*shop_target)], axis=1)
        # shop_cm = confusion_matrix(pred.shop, shop_target)
        # show_confustion_matrix(shop_cm)
        #
        # item_target = [label[3] for label in labels]
        # item_target = np.concatenate([list(b) for b in zip(*item_target)], axis=1)
        # item_cm = confusion_matrix(pred.item, item_target)
        # show_confustion_matrix(item_cm)
        #
        # scalar_target = [label[4] for label in labels]
        # scalar_target = np.concatenate([list(b) for b in zip(*scalar_target)], axis=1)
        # scalar_cm = confusion_matrix(pred.scalar, scalar_target)
        # show_confustion_matrix(scalar_cm)
