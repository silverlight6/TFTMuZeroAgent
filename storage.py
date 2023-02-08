import ray
import config
from Models.MuZero_agent_2 import TFTNetwork


@ray.remote
class Storage:
    def __init__(self, episode):
        self.target_model = self.load_model()
        if episode > 0:
            self.target_model.tft_load_model(episode)
        self.model = self.target_model
        self.episode_played = 0
        self.placements = {"player_" + str(r): [0 for _ in range(config.NUM_PLAYERS)]
                           for r in range(config.NUM_PLAYERS)}

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

    def record_placements(self, placement):
        print(placement)
        for key in self.placements.keys():
            # Increment which position each model got.
            self.placements[key][placement[key]] += 1
