import UnitTests.PlayerTests as PlayerTests
import UnitTests.MinionTests as MinionTests
import config


def runTest():
    if config.RUN_PLAYER_TESTS:
        PlayerTests.list_of_tests()
    if config.RUN_MINION_TESTS:
        MinionTests.list_of_tests()
