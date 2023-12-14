import UnitTests.player_test as PlayerTests
import UnitTests.minion_test as MinionTests
import UnitTests.drop_rate_test as DropRateTests
import UnitTests.MCTS_test as MCTSTest
import UnitTests.mapping_test as MappingTests
import UnitTests.checkpoint_test as CheckpointTests
import config


def runTest():
    if config.RUN_PLAYER_TESTS:
        PlayerTests.test_list()
    if config.RUN_MINION_TESTS:
        MinionTests.test_list()
    if config.RUN_DROP_TESTS:
        DropRateTests.test_list()
    if config.RUN_MCTS_TESTS:
        MCTSTest.test_list()
    if config.RUN_MAPPING_TESTS:
        MappingTests.test_list()
    if config.RUN_CHECKPOINT_TESTS:
        CheckpointTests.test_list()

