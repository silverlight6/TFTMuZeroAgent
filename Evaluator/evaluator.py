import config
import numpy as np
from Models.MCTS_torch import MCTS
from Models.MuZero_torch_agent import MuZeroNetwork as TFTNetwork

'''
Description -
    Loads in a set of agents from checkpoints of the users choice to play against each other. These agents all have
    their own policy function and are not required to be the same model. This method is used for evaluating the
    skill level of the current agent and to see how well our agents are training. Metrics from this method are 
    stored in the storage class because this is the data worker side so there are intended to be multiple copies
    of this method running at once. 
Inputs
    env 
        A parallel environment of our tft simulator. This is the game that the worker interacts with.
    storage
        An object that stores global information like the weights of the global model and current training progress
'''

def evaluate_agents(self, env, storage) -> None:
    agents = {"player_" + str(r): MCTS(TFTNetwork())
                for r in range(config.NUM_PLAYERS)}
    agents["player_1"].network.tft_load_model(1000)
    agents["player_2"].network.tft_load_model(2000)
    agents["player_3"].network.tft_load_model(3000)
    agents["player_4"].network.tft_load_model(4000)
    agents["player_5"].network.tft_load_model(5000)
    agents["player_6"].network.tft_load_model(6000)
    agents["player_7"].network.tft_load_model(7000)

    while True:
        # Reset the environment
        player_observation = env.reset()
        # This is here to make the input (1, observation_size) for initial_inference
        player_observation = np.asarray(
            list(player_observation.values()))
        # Used to know when players die and which agent is currently acting
        terminated = {
            player_id: False for player_id in env.possible_agents}
        # Current action to help with MuZero
        placements = {
            player_id: 0 for player_id in env.possible_agents}
        current_position = 7
        info = {player_id: {"player_won": False}
                for player_id in env.possible_agents}
        # While the game is still going on.
        while not all(terminated.values()):
            # Ask our model for an action and policy
            actions = {agent: 0 for agent in agents.keys()}
            for i, [key, agent] in enumerate(agents.items()):
                action, _ = agent.policy(np.expand_dims(player_observation[i], axis=0))
                actions[key] = action

            # step_actions = self.getStepActions(terminated, np.asarray(actions))

            # Take that action within the environment and return all of our information for the next player
            next_observation, reward, terminated, _, info = env.step(actions)

            # Set up the observation for the next action
            player_observation = np.asarray(list(next_observation.values()))

            for key, terminate in terminated.items():
                if terminate:
                    placements[key] = current_position
                    current_position -= 1

        for key, value in info.items():
            if value["player_won"]:
                placements[key] = 0
        storage.record_placements.remote(placements)
        print("recorded places {}".format(placements))
        self.rank += config.CONCURRENT_GAMES