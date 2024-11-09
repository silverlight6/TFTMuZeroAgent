import config
import numpy as np
import ray
import time
from Models.GumbelModels.gumbel_MCTS import GumbelMuZero
from Models.MuZero_torch_agent import MuZeroNetwork as TFTNetwork


@ray.remote(num_gpus=config.GPU_SIZE_PER_WORKER)
class SinglePlayerEvaluator(object):
    def __init__(self, model_config):
        # TODO: Move these to the individual methods so we don't need to build the project to run it.
        # Unless you are running these specific methods.

        self.temp_model = TFTNetwork(model_config)
        self.agent_network = GumbelMuZero(self.temp_model, model_config)
        self.past_network = GumbelMuZero(self.temp_model, model_config)

        # Testing purposes only
        self.live_game = True
        self.ckpt_time = time.time_ns()
        self.model_config = model_config

    def collect_single_player_experience(self, env, storage, weights, rank):
        self.agent_network.network.set_weights(weights)
        self.agent_network.network.eval()
        while True:
            # Reset the environment
            player_observation, info = env.vector_reset()
            player_observation = self.single_player_observation_to_input(player_observation)

            # Used to know when players die and which agent is currently acting
            terminated = [False for _ in range(env.num_envs)]
            storage_terminated = [False for _ in range(env.num_envs)]

            # While the game is still going on.
            while not all(terminated):
                # Ask our model for an action and policy. Use on normal case or if we only have current versions left
                actions, _, _ = self.agent_network.policy(player_observation)
                step_actions = self.getStepActions(terminated, actions, storage_terminated)
                for i, terminate in enumerate(terminated):
                    if terminate:
                        storage_terminated[i] = True
                        storage.record_game_result.remote(rank, info[0]["game_round"])

                # Take that action within the environment and return all of our information for the next player
                next_observation, reward, terminated, _, info = env.vector_step(step_actions, terminated)
                # store the action for MuZero
                # Using i for all possible players and alive_i for all alive players

                # Set up the observation for the next action
                player_observation = self.single_player_observation_to_input(next_observation)

    def single_player_observation_to_input(self, observation):
        scalars = []
        emb_scalars = []
        shop = []
        board = []
        bench = []
        items = []
        traits = []
        masks = []
        for obs in observation:
            scalars.append(obs["observations"]["scalars"])
            emb_scalars.append(obs["observations"]["emb_scalars"])
            shop.append(obs["observations"]["shop"])
            board.append(obs["observations"]["board"])
            bench.append(obs["observations"]["bench"])
            items.append(obs["observations"]["items"])
            traits.append(obs["observations"]["traits"])
            masks.append(obs["action_mask"])
        tensors = {
            "scalars": np.float32(np.expand_dims(np.array(scalars), 1)),
            "emb_scalars": np.array(emb_scalars),
            "shop": np.array(shop),
            "board": np.expand_dims(np.array(board), 1),
            "bench": np.expand_dims(np.array(bench), 1),
            "items": np.expand_dims(np.array(items), 1),
            "traits": np.expand_dims(np.array(traits), 1),
        }
        return {"observations": tensors,
                "action_mask": np.array(masks)}

    def getStepActions(self, terminated, actions, prev_terminations=None):
        step_actions = []
        i = 0
        for j, terminate in enumerate(terminated):
            if not terminate and i < len(actions):
                step_actions.append(actions[i])
                i += 1
            elif not terminate and i >= len(actions):
                step_actions.append(np.asarray(1976))
            elif terminate and not prev_terminations[j]:
                i += 1

        return step_actions

