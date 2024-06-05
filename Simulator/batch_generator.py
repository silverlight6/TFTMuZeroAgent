import config
import numpy as np

from Simulator.battle_generator import BattleGenerator
from Simulator import pool
from Simulator.observation.vector.observation import ObservationVector
from Simulator.observation.vector.gemini_observation import GeminiObservation
from Simulator.player_manager import PlayerManager
from Simulator.tft_simulator import TFTConfig


class BatchGenerator:
    def __init__(self):
        self.battle_generator = BattleGenerator()
        # self.observation_class = ObservationVector
        self.observation_class = GeminiObservation

    # So this needs to take in a batch size then generate the necessary number of positions
    # It will need to create the observations as the x and the y as the labels
    def generate_batch(self, batch_size):
        input_batch = []
        labels = []
        for _ in range(batch_size):
            starting_level = np.random.randint(1, 6)
            item_count = np.random.randint(0, 3)
            [player, opponent, other_players] = self.battle_generator.generate_battle(
                starting_level=starting_level, item_count=item_count, scenario_info=False, extra_randomness=False
            )

            pool_obj = pool.pool()
            player.opponent = opponent
            opponent.opponent = player

            player.shop = pool_obj.sample(player, 5)
            player.shop_champions = player.create_shop_champions()

            player.gold = np.random.randint(0, 102)
            player.exp = np.random.randint(0, player.level_costs[player.level])
            player.health = np.random.randint(1, 101)

            player_manager = PlayerManager(config.NUM_PLAYERS, pool_obj,
                                           TFTConfig(observation_class=self.observation_class))
            player_manager.reinit_player_set([player] + list(other_players.values()))

            initial_observation = player_manager.fetch_observation(f"player_{player.player_num}")
            observation = self.observation_class.observation_to_input(initial_observation)

            input_batch.append(observation)

            comp = player.get_tier_labels()
            champ = player.get_champion_labels()
            shop = player.get_shop_labels()
            item = player.get_item_labels()
            scalar = player.get_scalar_labels()

            labels.append([comp, champ, shop, item, scalar])
        input_batch = self.observation_class.observation_to_dictionary(input_batch)
        return input_batch, labels
