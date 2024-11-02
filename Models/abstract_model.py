import torch
import os

import config
from Models.MCTS_Util import dict_to_cpu

class AbstractNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def initial_inference(self, observation, training=False):
        pass

    def recurrent_inference(self, encoded_state, action, training=False):
        pass

    def get_weights(self) -> dict:
        return dict_to_cpu(self.state_dict())

    def set_weights(self, weights):
        self.load_state_dict(weights)
        self.eval()

    # Renaming as to not override built-in functions
    def tft_save_model(self, episode, optimizer):
        if not os.path.exists("./Checkpoints"):
            os.makedirs("./Checkpoints")

        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': episode,
        }

        path = f'./Checkpoints/checkpoint_{episode}'
        torch.save(checkpoint, path)

    # Renaming as to not override built-in functions
    def tft_load_model(self, episode):
        path = f'./Checkpoints/checkpoint_{episode}'
        if os.path.isfile(path):
            checkpoint = torch.load(path)
            self.load_state_dict(checkpoint['model_state_dict'])
            if config.TRAIN:
                self.train()
            else:
                self.eval()
            return checkpoint['optimizer_state_dict']
        # print("loaded model {}".format(episode))
