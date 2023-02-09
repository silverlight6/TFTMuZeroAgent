import AI_interface
import config
from UnitTests.BaseUnitTest import runTest
import argparse
import numpy as np
import tensorflow as tf


def main():
    if config.RUN_UNIT_TEST:
        runTest()

    # TODO(lobotuerk) A lot of hardcoded parameters should be used like this intead.
    parser = argparse.ArgumentParser(description='Train an AI to play TFT',
                                     epilog='For more information, '
                                            'go to https://github.com/silverlight6/TFTMuZeroAgent')
    parser.add_argument('--starting_episode', '-se', dest='starting_episode', type=int, default=0,
                        help='Episode number to start the training. '
                             'Used for loading checkpoints, disables loading if = 0')
    args = parser.parse_args()

    interface = AI_interface.AIInterface()
    # interface.train_model()
    interface.train_model(starting_train_step=args.starting_episode)
    # interface.collect_dummy_data()
    # interface.testEnv()
    # interface.PPO_algorithm()
    # interface.evaluate()


if __name__ == "__main__":
    main()
