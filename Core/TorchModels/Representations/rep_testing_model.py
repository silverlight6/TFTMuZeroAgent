import config
import torch
import torch.nn

from Core.TorchComponents.representation_models import RepPositionEmbeddingNetwork
from Core.TorchModels.abstract_model import AbstractNetwork


class RepresentationTesting(AbstractNetwork):
    def __init__(self, model_config):
        super().__init__()
        self.full_support_size = model_config.ENCODER_NUM_STEPS

        self.representation_network = RepPositionEmbeddingNetwork(model_config, use_round_count=False)

        self.prediction_network = PredNetwork(model_config)

        self.model_config = model_config

    def representation(self, observation):
        observation = {label: torch.from_numpy(value).to(config.DEVICE) for label, value in observation.items()}
        return self.representation_network(observation)

    def prediction(self, encoded_state):
        comp, champ = self.prediction_network(encoded_state)
        return comp, champ
        # comp, champ, shop, item, scalar = self.prediction_network(encoded_state)
        # return comp, champ, shop, item, scalar

    def forward(self, observation):
        hidden_state = self.representation(observation)
        comp, champ = self.prediction(hidden_state)
        # comp, champ, shop, item, scalar = self.prediction(hidden_state)

        outputs = {
            "comp": comp,
            "champ": champ
        }
        # outputs = {
        #     "comp": comp,
        #     "champ": champ,
        #     "shop": shop,
        #     "item": item,
        #     "scalar": scalar
        # }

        return outputs


class PredNetwork(torch.nn.Module):
    def __init__(self, model_config) -> torch.nn.Module:
        super().__init__()

        self.trait_outputs = torch.nn.ModuleList([
            torch.nn.Linear(model_config.HIDDEN_STATE_SIZE, tier_length) for tier_length in config.TEAM_TIERS_VECTOR
        ])

        self.champ_output = torch.nn.ModuleList([
            torch.nn.Linear(model_config.HIDDEN_STATE_SIZE, tier_length) for tier_length in config.CHAMPION_LIST_DIM
        ])
        #
        # self.shop_predictor_network = MultiMlp(model_config.HIDDEN_STATE_SIZE,
        #                                        [model_config.LAYER_HIDDEN_SIZE] * model_config.N_HEAD_HIDDEN_LAYERS,
        #                                        [63, 63, 63, 63, 63])
        #
        # self.item_predictor_network = MultiMlp(model_config.HIDDEN_STATE_SIZE,
        #                                        [model_config.LAYER_HIDDEN_SIZE] * model_config.N_HEAD_HIDDEN_LAYERS,
        #                                        [60 for _ in range(10)])
        #
        # self.scalar_predictor_network = MultiMlp(model_config.HIDDEN_STATE_SIZE,
        #                                          [model_config.LAYER_HIDDEN_SIZE] * model_config.N_HEAD_HIDDEN_LAYERS,
        #                                          [100 for _ in range(3)])

    def forward(self, hidden_state):
        trait_output = [head(hidden_state) for head in self.trait_outputs]
        champ_output = [head(hidden_state) for head in self.champ_output]

        return trait_output, champ_output
        # shop = self.shop_predictor_network(hidden_state)
        # item = self.item_predictor_network(hidden_state)
        # scalar = self.scalar_predictor_network(hidden_state)
        # return comp, champ, shop, item, scalar
