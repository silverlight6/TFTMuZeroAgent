import TestInterface.test_ai_interface as TestInterface
import argparse
import config
from TestInterface.test_atari import Atari_Tests
from Concurrency.AI_interface import AIInterface
from Evaluator.evaluator import Evaluator


def main():

    # TODO(lobotuerk) A lot of hardcoded parameters should be used like this instead.
    parser = argparse.ArgumentParser(description='Train an AI to play TFT',
                                     epilog='For more information, '
                                            'go to https://github.com/silverlight6/TFTMuZeroAgent')
    parser.add_argument('--starting_episode', '-se', dest='starting_episode', type=int,
                        help='Episode number to start the training. Used for loading checkpoints, '
                             'disables loading if = 0')

    parser.add_argument('--test', '-t', dest='test', action='store_true')

    args = parser.parse_args()

    if args.starting_episode:
        config.STARTING_EPISODE = args.starting_episode

    if args.test:
        test_interface = TestInterface.AIInterface()
        if config.MUZERO_POSITION:
            test_interface.train_position_model(starting_train_step=args.starting_episode)
        elif config.REP_TRAINER:
            test_interface.representation_testing()
            return
        test_interface.train_model(starting_train_step=args.starting_episode)
        return

    if config.EVALUATE:
        evaluator = Evaluator()
        evaluator.evaluate_single_player()
    elif config.ATARI:
        atari = Atari_Tests()
        atari.breakout()
    else:
        interface = AIInterface()
        if config.CHAMP_DECIDER:
            # interface.ppo_checkpoint_test()
            # interface.position_ppo_tune()
            # interface.train_guide_model()
            interface.position_ppo_testing()
        elif config.SINGLE_PLAYER:
            config.GUMBEL = True
            interface.train_single_player_model()
        elif config.REP_TRAINER:
            # interface.representation_testing()
            interface.representation_evauation()
        elif config.MUZERO_POSITION:
            interface.train_position_model()
        else:
            interface.train_torch_model()


if __name__ == "__main__":
    main()
