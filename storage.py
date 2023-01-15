import ray
from Models.MuZero_agent_2 import TFTNetwork


@ray.remote
class Storage:
    def __init__(self, episode):
        self.target_model = self.load_model()
        #self.target_model.tft_load_model(episode)
        self.model = self.target_model
        self.episode_played = 0

    def get_model(self):
        return self.model.get_weights()

    def set_model(self):
        self.model.set_weights(self.target_model.get_weights())

    # Implementing saving.
    def load_model(self):
        return TFTNetwork()

    def get_target_model(self):
        return self.target_model.get_weights()

    def set_target_model(self, weights):
        return self.target_model.set_weights(weights)

    def get_episode_played(self):
        return self.episode_played

    def increment_episode_played(self):
        self.episode_played += 1

