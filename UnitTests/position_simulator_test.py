import config
import numpy as np
from Simulator.tft_position_simulator import TFT_Position_Simulator


def create_simulator():
    # Create a simulator instance
    sim = TFT_Position_Simulator()
    sim.reset()  # Ensure the environment is reset for testing
    return sim

def observation_changes_after_action_test(simulator):
    config.MUZERO_POSITION = True
    config.PRESET_BATTLE = True
    # Get the initial observation
    initial_observation, _ = simulator.reset()

    # Verify that there is no unit at the specified position [3, 0]
    # For simplicity, we assume `player.board` can be accessed to check positions.
    # Replace `board_access_method` with the actual way to access player's board.
    assert simulator.PLAYER.board[0][3] is None, "Expected no unit at position [3, 0] initially."

    # Define the action, here 0 for simplicity
    action = np.array(0, dtype=int)

    # Step the environment with the action
    new_observation, reward, terminated, truncated, info = simulator.step(action)

    # Check that the new observation is different from the initial observation
    assert not np.array_equal(
        initial_observation["observations"]["board"], new_observation["observations"]["board"]
    ), "Expected observation to change after action was taken."

def test_list():
    observation_changes_after_action_test(create_simulator())
