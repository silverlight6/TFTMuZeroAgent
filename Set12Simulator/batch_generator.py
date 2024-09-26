import config
import numpy as np

from Set12Simulator.battle_generator import BattleGenerator
from Set12Simulator import pool
from Set12Simulator.observation.token.basic_observation import ObservationToken
from Set12Simulator.observation.vector.observation import ObservationVector
from Set12Simulator.observation.vector.gemini_observation import GeminiObservation
from Set12Simulator.player_manager import PlayerManager
from Set12Simulator.tft_simulator import TFTConfig

class BatchGenerator:
    def __init__(self):
        # TODO: Find the bug with 5 costs that is causing instability in the add to bench, most likely kayn
        # TODO: Another bug if you add items.. Don't have the energy to find it today.
        base_level_config = {
            "num_unique_champions": 12,
            "max_cost": 5,
            "num_items": 0,
            "current_level": 8,
            "chosen": True,
            "sample_from_pool": True,
            "two_star_unit_percentage": .3,
            "three_star_unit_percentage": .01,
            "scenario_info": True,
            "extra_randomness": False
        }

        self.battle_generator = BattleGenerator(base_level_config)
        self.observation_class = ObservationToken
        # self.observation_class = GeminiObservation

    # So this needs to take in a batch size then generate the necessary number of positions
    # It will need to create the observations as the x and the y as the labels
    def generate_batch(self, batch_size):
        input_batch = []
        labels = []
        for _ in range(batch_size):
            [player, opponent, other_players] = self.battle_generator.generate_battle()

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

            initial_observation = player_manager.fetch_position_observation(f"player_{player.player_num}")
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
