import TestInterface.test_ai_interface as TestInterface
import config
from UnitTests.BaseUnitTest import runTest
import argparse
import AI_interface


def main():
    if config.RUN_UNIT_TEST:
        runTest()

    # TODO(lobotuerk) A lot of hardcoded parameters should be used like this instead.
    parser = argparse.ArgumentParser(description='Train an AI to play TFT',
                                     epilog='For more information, '
                                            'go to https://github.com/silverlight6/TFTMuZeroAgent')
    parser.add_argument('--starting_episode', '-se', dest='starting_episode', type=int, default=0,
                        help='Episode number to start the training. Used for loading checkpoints, '
                             'disables loading if = 0')

    parser.add_argument('--test', '-t', dest='test', action='store_true')
    args = parser.parse_args()

    if args.test:
        test_interface = TestInterface.AIInterface()
        test_interface.train_model(starting_train_step=args.starting_episode)
        return

    interface = AI_interface.AIInterface()
    interface.train_torch_model(starting_train_step=args.starting_episode)
    # interface.collect_dummy_data()
    # interface.testEnv()
    # interface.PPO_algorithm()
    while True:
        try:
            test_interface = TestInterface.AIInterface()
            test_interface.evaluate(0)
        except:
            # print("OK")
            pass
    test_interface.train_model(starting_train_step=args.starting_episode)


if __name__ == "__main__":
    main()
