# So this is the test that I want to check if the same information is being delivered back to the source.
import config
from unittest.mock import MagicMock, call
from Concurrency.data_worker import DataWorker
def collect_position_experience_test():
    # Create mock environment, buffers, and storage
    mock_env = MagicMock()
    mock_buffers = MagicMock()
    mock_global_buffer = MagicMock()
    mock_storage = MagicMock()

    # Initialize environment properties
    mock_env.num_envs = 3
    mock_env.vector_reset.return_value = ([{"observations": [0, 1, 2]}], [{"num_units": 3}])
    mock_env.vector_step.side_effect = [
        ([{"observations": [1, 2, 3]}], [1.0], [False, False, False], [None], [{"num_units": 3}]),
        ([{"observations": [2, 3, 4]}], [0.5], [True, True, False], [None], [{"num_units": 2}]),
    ]

    # Setup agent network with mocked methods
    mock_agent_network = MagicMock()
    mock_agent_network.policy.return_value = ([0, 1, 2], [0.3, 0.4, 0.5], [0.6, 0.7, 0.8])

    model_config = config.ModelConfig()
    # Create the test instance of your class and inject dependencies
    test_instance = DataWorker.remote(0, model_config)  # Replace with actual class name
    test_instance.agent_network = mock_agent_network

    # Set buffer size limit for testing buffer overflow
    config.GLOBAL_BUFFER_SIZE = 10  # Use a small buffer size for testing

    # Run the method to gather data
    test_instance.collect_position_experience.remote(mock_env, mock_buffers, mock_global_buffer,
                                                     mock_storage, weights=None)

    # Assertions for buffer storing behavior
    # Check that local buffer was called with expected values
    local_calls = [
        call("player_0", [0, 1, 2], 0, 1.0, 0.3, 0.6),
        call("player_1", [0, 1, 2], 1, 1.0, 0.4, 0.7),
        call("player_2", [0, 1, 2], 2, 1.0, 0.5, 0.8),
        # Continue for each step in the game
    ]
    mock_buffers.store_gumbel_buffer.remote.assert_has_calls(local_calls, any_order=True)

    # Check that global buffer is updated at game end
    mock_buffers.store_global_position_buffer.remote.assert_called_once_with(mock_global_buffer)

    # Verify global buffer sampling maintains consistency
    sampled_data = mock_global_buffer.sample_data()
    expected_data = [
        {"observation": [0, 1, 2], "reward": 1.0, "action": 0, "policy": 0.3, "value": 0.6},
        {"observation": [1, 2, 3], "reward": 0.5, "action": 1, "policy": 0.4, "value": 0.7},
        # Add additional expected data for complete game
    ]
    assert sampled_data == expected_data, "Sampled data does not match expected sequence."

    # Additional assertion for global buffer overflow control
    mock_global_buffer.buffer_size.assert_called()
    assert mock_global_buffer.buffer_size() <= config.GLOBAL_BUFFER_SIZE * 0.8, "Global buffer overflow detected."

def test_list():
    collect_position_experience_test()