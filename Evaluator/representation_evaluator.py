import config
import numpy as np
import time
import torch

from Evaluator.eval_visualizers import display_confusion_matrices, show_confustion_matrix
from Simulator.batch_generator import BatchGenerator
from Simulator.origin_class_stats import tiers
from Simulator.stats import COST
from Simulator.tft_config import TFTConfig
from sklearn.metrics import confusion_matrix

class RepresentationEvaluator:
    def __init__(self, global_agent):
        self.network = global_agent
        tftConfig = TFTConfig()
        from Simulator.observation.token.basic_observation import ObservationToken
        tftConfig.observation_class = ObservationToken
        self.batch_generator = BatchGenerator(tftConfig)
        self.softmax = torch.nn.Softmax(dim=-1)

    def evaluate(self):
        observation, labels = self.batch_generator.generate_batch(batch_size=config.BATCH_SIZE)
        predictions = self.compute_forward(observation)

        self.create_graphs(predictions, labels)

    def compute_forward(self, observation):
        self.network.train()
        output = self.network.forward(observation)

        # predictions = Prediction(
        #     comp=comp_output_list.numpy(),
        #     champ=torch.argmax(torch.cat([t.flatten() for t in output["champ"]]), -1).to(dtype=torch.int8).cpu(),
        #     shop=torch.argmax(torch.cat([t.flatten() for t in output["shop"]]), -1).to(dtype=torch.int8).cpu(),
        #     item=torch.argmax(torch.cat([t.flatten() for t in output["item"]]), -1).to(dtype=torch.int8).cpu(),
        #     scalar=torch.argmax(torch.cat([t.flatten() for t in output["scalar"]]), -1).to(dtype=torch.int8).cpu()
        # )
        return output

    def create_graphs(self, pred, labels):
        self.create_comp_graph(pred["comp"], [label[0] for label in labels])

        self.create_champion_graph(pred["champ"], [label[1] for label in labels])

        # self.create_shop_graph(pred["shop"], [label[2] for label in labels])
        #
        # self.create_item_graph(pred["item"], [label[3] for label in labels])
        #
        # self.create_scalar_graph(pred["scalar"], [label[4] for label in labels])

    def temp_create_comp_graph(self, pred, labels):
        print(pred)
        comp_output_list = torch.tensor([])
        for t in pred:
            max_index = torch.argmax(self.softmax(t))
            comp_output_list = torch.cat((comp_output_list, torch.unsqueeze(max_index, dim=0).detach().cpu()))

        label_output = [[] for _ in range(len(config.TEAM_TIERS_VECTOR))]
        for label in labels:
            for i, comp in enumerate(label):
                max_index = np.argmax(comp)
                label_output[i] = np.concatenate([label_output[i], [max_index]], axis=0)

        prediction = [comp.numpy() for comp in comp_output_list]
        divine_label = label_output[1]
        print(f"prediction {prediction}, label {divine_label}")

        cm = confusion_matrix(divine_label, prediction)
        show_confustion_matrix(cm)

    # So I want to create a graph that has the names of each of the classes from the tiers list.
    def create_comp_graph(self, pred, labels):
        pred, tier_target = self.process_tier_comp(pred, labels, len(config.TEAM_TIERS_VECTOR))

        confusion_matrices = [confusion_matrix(tier_target[i], pred[i]) for i in range(len(tier_target))]

        # Display confusion matrices in a scrollable window
        display_confusion_matrices(confusion_matrices, list(tiers.keys()))

    def create_champion_graph(self, pred, labels):
        pred, tier_target = self.process_tier_comp(pred, labels, len(config.CHAMPION_ACTION_DIM))
        confusion_matrices = [confusion_matrix(tier_target[i], pred[i]) for i in range(len(tier_target))]

        # Display confusion matrices in a scrollable window
        display_confusion_matrices(confusion_matrices, list(COST.keys())[1:])

    def process_tier_comp(self, output, labels, dimensions):
        comp_output_list = [torch.tensor([]) for _ in range(dimensions)]
        for i, t in enumerate(output):
            for comp in t:
                max_index = torch.argmax(self.softmax(comp))
                comp_output_list[i] = torch.cat((comp_output_list[i], torch.unsqueeze(max_index, dim=0).detach().cpu()))

        label_output = [[] for _ in range(dimensions)]
        for label in labels:
            for i, comp in enumerate(label):
                max_index = np.argmax(comp)
                label_output[i] = np.concatenate([label_output[i], [max_index]], axis=0)

        return [comp_list.numpy() for comp_list in comp_output_list], label_output

    def create_shop_graph(self, pred, labels):
        shop_target = [label[2] for label in labels]
        shop_target = np.concatenate([list(b) for b in zip(*shop_target)], axis=1)
        shop_cm = confusion_matrix(pred.shop, shop_target)
        show_confustion_matrix(shop_cm)

    def create_item_graph(self, pred, labels):
        item_target = [label[3] for label in labels]
        item_target = np.concatenate([list(b) for b in zip(*item_target)], axis=1)
        item_cm = confusion_matrix(pred.item, item_target)
        show_confustion_matrix(item_cm)

    def create_scalar_graph(self, pred, labels):
        scalar_target = [label[4] for label in labels]
        scalar_target = np.concatenate([list(b) for b in zip(*scalar_target)], axis=1)
        scalar_cm = confusion_matrix(pred.scalar, scalar_target)
        show_confustion_matrix(scalar_cm)