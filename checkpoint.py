from Models.MuZero_torch_agent import MuZeroNetwork as TFTNetwork

class Checkpoint:
    def __init__(self, epoch, q_score):
        self.epoch = epoch
        self.q_score = q_score

    def get_model(self):
        model = TFTNetwork()
        if self.epoch == 0:
            return model.get_weights()
        else:
            model.tft_load_model(self.epoch)
            return model.get_weights()

    def update_q_score(self, episode, prob):
        if episode != 0:
            self.q_score = self.q_score - (0.01 / (episode * prob))
        else:
            self.q_score = self.q_score - 0.01 / prob