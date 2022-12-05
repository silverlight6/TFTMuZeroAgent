import AI_interface
import config
from UnitTests.BaseUnitTest import runTest


def main():
    if config.RUN_UNIT_TEST:
        runTest()

    AI_interface.train_model()


if __name__ == "__main__":
    main()
pass