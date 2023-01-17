import ray
import config
from Models.MuZero_agent_2 import TFTNetwork
from Models.A3C_Agent import A3C_Agent


@ray.remote
class Storage:
    def __init__(self, episode):
        self.target_model = self.load_model()
        if episode > 0:
            self.target_model.tft_load_model(episode)
        self.model = self.target_model
        self.episode_played = 0

    def get_model(self):
        if config.MODEL == "MuZero":
            return self.model.get_weights()
        elif config.MODEL == "A3C":
            return self.model.a3c_net.get_weights()
        else:
            return self.model.get_weights()

    def set_model(self):
        if config.MODEL == "MuZero":
            self.model.set_weights(self.target_model.get_weights())
        elif config.MODEL == "A3C":
            self.model.a3c_net.set_weights(self.target_model.a3c_net.get_weights())
        else:
            self.model.set_weights(self.target_model.get_weights())

    # Implementing saving.
    def load_model(self):
        if config.MODEL == "MuZero":
            return TFTNetwork()
        elif config.MODEL == "A3C":
            return A3C_Agent(config.INPUT_SHAPE)
        else:
            return TFTNetwork()

    def get_target_model(self):
        if config.MODEL == "MuZero":
            return self.target_model.get_weights()
        elif config.MODEL == "A3C":
            return self.target_model.a3c_net.get_weights()
        else:
            return self.target_model.get_weights()

    def set_target_model(self, weights):
        if config.MODEL == "MuZero":
            self.target_model.set_weights(weights)
        elif config.MODEL == "A3C":
            self.target_model.a3c_net.set_weights(weights)
        else:
            self.target_model.set_weights(weights)

    def get_episode_played(self):
        return self.episode_played

    def increment_episode_played(self):
        self.episode_played += 1

