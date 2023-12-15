import numpy as np
from os import environ, path
from dotenv import load_dotenv

BASE_DIR = path.abspath(path.join(path.dirname(__file__), "."))
load_dotenv(path.join(BASE_DIR, ".env"))

GPU_SIZE_PER_WORKER = environ.get("GPU_SIZE_PER_WORKER", 0.2)
STORAGE_GPU_SIZE = environ.get("STORAGE_GPU_SIZE", 0.1)
BUFFER_GPU_SIZE = environ.get("BUFFER_GPU_SIZE", 0.02)
TRAINER_GPU_SIZE = environ.get("TRAINER_GPU_SIZE", 0.2)

# IMPORTANT: Change this value to the number of cpu cores you want to use (recommended 80% of cpu)
NUM_CPUS = environ.get("NUM_CPUS", 14)

DEVICE = environ.get("DEVICE", "cuda")
IMITATION = environ.get("IMITATION", False)
CHAMP_DECIDER = environ.get("CHAMP_DECIDER", False)

### TIME RELATED VALUES ###
ACTIONS_PER_TURN = environ.get("ACTIONS_PER_TURN", 15)
CONCURRENT_GAMES = environ.get("CONCURRENT_GAMES", 2)
NUM_PLAYERS = environ.get("NUM_PLAYERS", 8)

AUTO_BATTLER_PERCENTAGE = environ.get("AUTO_BATTLER_PERCENTAGE", 0)

# Putting this here so that we don't scale the policy by a multiple of 5
# Because we calculate the loss for each of the 5 dimensions.
# I'll add a mathematical way of generating these numbers later.
DEBUG = environ.get("DEBUG", False)
CHECKPOINT_STEPS = environ.get("CHECKPOINT_STEPS", 100)

STARTING_EPISODE = environ.get("STARTING_EPISODE", 0)

CHAMPION_ACTION_DIM = [5 for _ in range(58)]
CHAMPION_LIST_DIM = [2 for _ in range(58)]
ITEM_CHOICE_DIM = [3 for _ in range(10)]
CHAMP_DECIDER_ACTION_DIM = CHAMPION_ACTION_DIM + [2] + ITEM_CHOICE_DIM

DISCOUNT = environ.get("DISCOUNT", 0.997)
TRAINING_STEPS = environ.get("TRAINING_STEPS", 1e10)
OBSERVATION_SIZE = environ.get("OBSERVATION_SIZE", 10432)
OBSERVATION_TIME_STEPS = environ.get("OBSERVATION_TIME_STEPS", 4)
OBSERVATION_TIME_STEP_INTERVAL = environ.get("OBSERVATION_TIME_STEP_INTERVAL", 5)
INPUT_TENSOR_SHAPE = np.array([OBSERVATION_SIZE])
ACTION_ENCODING_SIZE = environ.get("ACTION_ENCODING_SIZE", 1045)
ACTION_CONCAT_SIZE = environ.get("ACTION_CONCAT_SIZE", 81)
ACTION_DIM = [7, 37, 10]
# 57 is the number of champions in set 4. Don't want to add an import to the STATS in the simulator in a config file


# Number of categories for each trait tier. Emperor for example has 2, no emperors or 1.
TEAM_TIERS_VECTOR = [4, 5, 4, 4, 4, 3, 3, 3, 2, 4, 4, 4, 5, 3, 5, 2, 3, 5, 4, 4, 3, 4, 4, 4, 2, 5]
TIERS_FLATTEN_LENGTH = 97
CHANCE_BUFFER_SEND = environ.get("CHANCE_BUFFER_SEND", 1)
GLOBAL_BUFFER_SIZE = environ.get("GLOBAL_BUFFER_SIZE", 20000)
ITEM_POSITIONING_BUFFER_SIZE = environ.get("ITEM_POSITIONING_BUFFER_SIZE", 2000)
MINIMUM_POP_AMOUNT = environ.get("MINIMUM_POP_AMOUNT", 100)

# Set to -1 to turn off.
TD_STEPS = environ.get("TD_STEPS", -1)

# This should be 1000 + because we want to be sampling everything when using priority.
# To change, look into the code in replay_muzero_buffer
SAMPLES_PER_PLAYER = environ.get("SAMPLES_PER_PLAYER", 1000)

# For default agent, this needs to be low because there often isn't many samples per game.
UNROLL_STEPS = environ.get("UNROLL_STEPS", 5)

### TRAINING ###
BATCH_SIZE = environ.get("BATCH_SIZE", 1024)
INIT_LEARNING_RATE = environ.get("INIT_LEARNING_RATE", 0.01)
LEARNING_RATE_DECAY = environ.get("LEARNING_RATE_DECAY", int(350e3))
LR_DECAY_FUNCTION = environ.get("LR_DECAY_FUNCTION", 0.1)
WEIGHT_DECAY = environ.get("WEIGHT_DECAY", 1e-5)
REWARD_LOSS_SCALING = environ.get("REWARD_LOSS_SCALING", 1)
POLICY_LOSS_SCALING = environ.get("POLICY_LOSS_SCALING", 1)
VALUE_LOSS_SCALING = environ.get("VALUE_LOSS_SCALING", 1)
GAME_METRICS_SCALING = environ.get("GAME_METRICS_SCALING", 0.2)

# INPUT SIZES
SHOP_INPUT_SIZE = 45
BOARD_INPUT_SIZE = 728
BENCH_INPUT_SIZE = 234
STATE_INPUT_SIZE = 85
COMP_INPUT_SIZE = 102
OTHER_PLAYER_INPUT_SIZE = 5180
OTHER_PLAYER_ITEM_POS_SIZE = 5920

OBSERVATION_LABELS = ["shop", "board", "bench", "states", "game_comp", "other_players"]
POLICY_HEAD_SIZES = [7, 5, 630, 370, 9]  # [7 types, shop, movement, item, sell/item loc]
NEEDS_2ND_DIM = [1, 2, 3, 4]

MINIMUM_REWARD = environ.get("MINIMUM_REWARD", -300)
MAXIMUM_REWARD = environ.get("MAXIMUM_REWARD", 300.0)

class ModelConfig:
    # AI RELATED VALUES START HERE

    #### MODEL SET UP ####
    HIDDEN_STATE_SIZE = environ.get("HIDDEN_STATE_SIZE", 256)
    NUM_RNN_CELLS = environ.get("NUM_RNN_CELLS", 2)
    LSTM_SIZE = int(HIDDEN_STATE_SIZE / (NUM_RNN_CELLS * 2))
    RNN_SIZES = [LSTM_SIZE] * NUM_RNN_CELLS
    LAYER_HIDDEN_SIZE = environ.get("LAYER_HIDDEN_SIZE", 256)
    ROOT_DIRICHLET_ALPHA = environ.get("ROOT_DIRICHLET_ALPHA", 1.0)
    ROOT_EXPLORATION_FRACTION = environ.get("ROOT_EXPLORATION_FRACTION", 0.25)
    VISIT_TEMPERATURE = environ.get("VISIT_TEMPERATURE", 1.0)

    PB_C_BASE = environ.get("PB_C_BASE", 19652)
    PB_C_INIT = environ.get("PB_C_INIT", 1.25)

    # ACTION_DIM = 10
    ENCODER_NUM_STEPS = environ.get("ENCODER_NUM_STEPS", 601)
    SELECTED_SAMPLES = environ.get("SELECTED_SAMPLES", True)
    MAX_GRAD_NORM = environ.get("MAX_GRAD_NORM", 5)

    N_HEAD_HIDDEN_LAYERS = 2

    NUM_SAMPLES = environ.get("NUM_SAMPLES", 30)
    NUM_SIMULATIONS = environ.get("NUM_SIMULATIONS", 50)
