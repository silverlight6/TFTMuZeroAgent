import UnitTests.PlayerTests as PlayerTests
import config


def runTest():
    if config.RUN_PLAYER_TESTS:
        PlayerTests.list_of_tests()
