import UnitTests.PlayerTests as PlayerTests
import UnitTests.MinionTests as MinionTests
import UnitTests.DropRateTests as DropRateTests
import UnitTests.MCTSTest as MCTSTest
import config


def runTest():
    if config.RUN_PLAYER_TESTS:
        PlayerTests.list_of_tests()
    if config.RUN_MINION_TESTS:
        MinionTests.list_of_tests()
    if config.RUN_DROP_TESTS:
        DropRateTests.list_of_tests()
    if config.RUN_MCTS_TESTS:
        MCTSTest.list_of_tests()
