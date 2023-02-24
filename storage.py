import ray
import config


@ray.remote(num_gpus=0.01)
class Storage:
    def __init__(self, episode):
        self.target_model = self.load_model()
        if episode > 0:
            self.target_model.tft_load_model(episode)
        self.model = self.target_model
        self.episode_played = 0
        self.placements = {"player_" + str(r): [0 for _ in range(config.NUM_PLAYERS)]
                           for r in range(config.NUM_PLAYERS)}
        self.trainer_busy = False

    def get_model(self):
        return self.model.get_weights()

    def set_model(self):
        self.model.set_weights(self.target_model.get_weights())

    # Implementing saving.
    def load_model(self):
        if config.ARCHITECTURE == "Tensorflow":
            from Models.MuZero_keras_agent import TFTNetwork
        else:
            from Models.MuZero_torch_agent import MuZeroNetwork as TFTNetwork
        return TFTNetwork()

    def get_target_model(self):
        return self.target_model.get_weights()

    def set_target_model(self, weights):
        return self.target_model.set_weights(weights)

    def get_episode_played(self):
        return self.episode_played

    def increment_episode_played(self):
        self.episode_played += 1

    def set_trainer_busy(self, status):
        self.trainer_busy = status

    def get_trainer_busy(self):
        return self.trainer_busy

    def record_placements(self, placement):
        print(placement)
        for key in self.placements.keys():
            # Increment which position each model got.
            self.placements[key][placement[key]] += 1
