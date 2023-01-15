import AI_interface
import config
from UnitTests.BaseUnitTest import runTest


def main():
    if config.RUN_UNIT_TEST:
        runTest()

    interface = AI_interface.AIInterface()
    interface.train_model()
    #interface.test_train_model()
    # interface.collect_dummy_data()
    # interface.testEnv()
    # interface.PPO_algorithm()


if __name__ == "__main__":
    main()
