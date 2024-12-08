import config
from Core.TorchModels.MuZero_torch_agent import MuZeroNetwork as TFTNetwork
from Core.TorchModels.Muzero_default_agent import MuZeroDefaultNetwork as DefaultNetwork
from Core.TorchModels.Representations.rep_testing_model import RepresentationTesting as RepNetwork


# TODO: Add description / inputs when doing unit testing on this object
"""
Description - 
Inputs      -
"""
class Checkpoint:
    def __init__(self, epoch, q_score, model_config):
        self.epoch = epoch
        self.q_score = q_score
        self.model_config = model_config

    # TODO: Add description / outputs when doing unit testing on this method
    """
    Description - 
    Outputs     - 
    """
    def get_model(self) -> dict:
        if config.CHAMP_DECIDER:
            model = DefaultNetwork(self.model_config)
        elif config.REP_TRAINER:
            model = RepNetwork(self.model_config)
        else:
            model = TFTNetwork(self.model_config)
        if self.epoch == 0:
            return model.get_weights()
        else:
            model.tft_load_model(self.epoch)
            return model.get_weights()

    # TODO: Add description / inputs when doing unit testing on this method
    """
    Description - 
    Inputs      - 
    """
    def update_q_score(self, episode, prob) -> None:
        if episode != 0:
            self.q_score = self.q_score - (0.01 / (episode * prob))
        else:
            self.q_score = self.q_score - 0.01 / prob
        # setting a lower limit, so it's possible that it will get sampled at some small number
        if self.q_score < 0.001:
            self.q_score = 0.001
