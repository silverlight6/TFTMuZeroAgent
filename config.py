import numpy as np
from os import environ, path, cpu_count
from dotenv import load_dotenv

# Load environment variables from .env file
BASE_DIR = path.abspath(path.join(path.dirname(__file__), "."))
load_dotenv(path.join(BASE_DIR, ".env"))

def get_bool_env(var_name, default="False"):
    """Converts environment variable to a boolean."""
    return environ.get(var_name, default).lower() in ("true", "1", "t")


def get_float_env(var_name, default=0.0):
    """Fetches an environment variable and converts it to float."""
    return float(environ.get(var_name, default))


def get_int_env(var_name, default=0):
    """Fetches an environment variable and converts it to int."""
    return int(environ.get(var_name, default))

def get_num_cpus(percentage=0.8):
    """Calculates the number of CPU cores to use based on a percentage of the total available cores.
    
    Args:
        percentage (float): The percentage of total cores to use (80% by default).

    Returns:
        int: The calculated number of cores to use, or the value from NUM_CPUS environment variable if set.
    """
    max_cpus = cpu_count() or 1  # Use at least 1 core if os.cpu_count() returns None
    default_cores = max(1, int(max_cpus * percentage))  # Calculate default, ensure at least 1
    
    # Fetch NUM_CPUS from env, falling back to calculated default_cores if not specified
    num_cpus = get_int_env("NUM_CPUS", default_cores)
    
    # Ensure NUM_CPUS does not exceed the actual number of available cores
    return min(num_cpus, max_cpus)

# IMPORTANT: Change this value to the percentage of CPU cores you want to use (default 80%)
NUM_CPUS = get_num_cpus()

# try:
#     import torch
#     # Dynamically set the device based on available hardware acceleration
#     if torch.cuda.is_available():
#         DEVICE = "cuda"
#     elif torch.backends.mps.is_available(): # Apple Silicon
#         DEVICE = "mps"
#     else:
#         DEVICE = "cpu"
# except ImportError:
#     print("WARNING: Torch import failed, may not be installed, AI capabilities not available.")
DEVICE = "cuda"

IMITATION = get_bool_env("IMITATION")
CHAMP_DECIDER = get_bool_env("CHAMP_DECIDER")
REP_TRAINER = get_bool_env("REP_TRAINER")

PATH = path.dirname(path.realpath(__file__))

# GPU Configurations
GPU_SIZE_PER_WORKER = get_float_env("GPU_SIZE_PER_WORKER", 0.2) if DEVICE == "cuda" else 0
STORAGE_GPU_SIZE = get_float_env("STORAGE_GPU_SIZE", 0.1) if DEVICE == "cuda" else 0
BUFFER_GPU_SIZE = get_float_env("BUFFER_GPU_SIZE", 0.02) if DEVICE == "cuda" else 0
TRAINER_GPU_SIZE = get_float_env("TRAINER_GPU_SIZE", 0.5) if DEVICE == "cuda" else 0

### TIME RELATED VALUES ###
ACTIONS_PER_TURN = get_int_env("ACTIONS_PER_TURN", 15)
CONCURRENT_GAMES = get_int_env("CONCURRENT_GAMES", 1)
NUM_PLAYERS = get_int_env("NUM_PLAYERS", 8)
AUTO_BATTLER_PERCENTAGE = get_int_env("AUTO_BATTLER_PERCENTAGE", 0)
DEBUG = get_bool_env("DEBUG", "True")
CHECKPOINT_STEPS = get_int_env("CHECKPOINT_STEPS", 100)
STARTING_EPISODE = get_int_env("STARTING_EPISODE", 0)

# Champion and Item Dimensions
CHAMPION_ACTION_DIM = [5 for _ in range(58)]
CHAMPION_LIST_DIM = [2 for _ in range(58)]
ITEM_CHOICE_DIM = [3 for _ in range(10)]
CHAMP_DECIDER_ACTION_DIM = CHAMPION_ACTION_DIM + [2] + ITEM_CHOICE_DIM

# Team Tiers Vector - Number of categories for each trait tier. Emperor, for example, has 2: no emperors or 1.
TEAM_TIERS_VECTOR = [
    4, 5, 4, 4, 4, 3, 3, 3, 2, 4,
    4, 4, 5, 3, 5, 2, 3, 5, 4, 4,
    3, 4, 4, 4, 2, 5,
]
TIERS_FLATTEN_LENGTH = 97

DISCOUNT = get_float_env("DISCOUNT", 0.997)
TRAINING_STEPS = 1e10
OBSERVATION_SIZE = get_int_env("OBSERVATION_SIZE", 10432)
OBSERVATION_TIME_STEPS = get_int_env("OBSERVATION_TIME_STEPS", 4)
OBSERVATION_TIME_STEP_INTERVAL = get_int_env("OBSERVATION_TIME_STEP_INTERVAL", 5)
INPUT_TENSOR_SHAPE = np.array([OBSERVATION_SIZE])
CHAMP_ENCODING_SIZE = get_int_env("CHAMP_ENCODING_SIZE", 26)
ACTION_ENCODING_SIZE = get_int_env("ACTION_ENCODING_SIZE", 1045)
ACTION_CONCAT_SIZE = get_int_env("ACTION_CONCAT_SIZE", 83)
ACTION_DIM = [7, 38, 38]

# Buffer settings and sample management
CHANCE_BUFFER_SEND = get_int_env("CHANCE_BUFFER_SEND", 1)
GLOBAL_BUFFER_SIZE = get_int_env("GLOBAL_BUFFER_SIZE", 20000)
ITEM_POSITIONING_BUFFER_SIZE = get_int_env("ITEM_POSITIONING_BUFFER_SIZE", 4000)
MINIMUM_POP_AMOUNT = get_int_env("MINIMUM_POP_AMOUNT", 100)

# Temporal Difference steps - Set to -1 to turn off.
TD_STEPS = get_int_env("TD_STEPS", -1)

# Sampling priority configuration - Aim for 1000+ to ensure comprehensive sampling.
SAMPLES_PER_PLAYER = get_int_env("SAMPLES_PER_PLAYER", 1000)

# Unroll steps for the default agent - Keep low due to potential scarcity of samples.
UNROLL_STEPS = get_int_env("UNROLL_STEPS", 5)

### TRAINING CONFIGURATIONS ###
BATCH_SIZE = get_int_env("BATCH_SIZE", 1024)
INIT_LEARNING_RATE = get_float_env("INIT_LEARNING_RATE", 0.01)
LR_DECAY_FUNCTION = get_float_env("LR_DECAY_FUNCTION", 0.1)
WEIGHT_DECAY = get_float_env("WEIGHT_DECAY", 1e-5)
DECAY_STEPS = get_int_env("DECAY_STEPS", 100)
REWARD_LOSS_SCALING = get_int_env("REWARD_LOSS_SCALING", 1)
POLICY_LOSS_SCALING = get_int_env("POLICY_LOSS_SCALING", 1)
VALUE_LOSS_SCALING = get_int_env("VALUE_LOSS_SCALING", 1)
GAME_METRICS_SCALING = get_float_env("GAME_METRICS_SCALING", 0.2)

# Input dimensions for different parts of the observation space
SCALAR_INPUT_SIZE = 76
SHOP_INPUT_SIZE = 45
BOARD_INPUT_SIZE = 728
BENCH_INPUT_SIZE = 234
ITEMS_INPUT_SIZE = 60
TRAIT_INPUT_SIZE = 102
OTHER_PLAYER_INPUT_SIZE = 5866
OTHER_PLAYER_POS_INPUT_SIZE = 5810
OTHER_PLAYER_ITEM_POS_SIZE = 5920
OTHER_PLAYER_SCALAR_SIZE = 8

# Observation labels and dimensions
OBSERVATION_LABELS = ["shop", "board", "bench", "states", "game_comp", "other_players"]
POLICY_HEAD_SIZE = 2090
NEEDS_2ND_DIM = [1, 2, 3, 4]

# Reward boundaries
MINIMUM_REWARD = get_float_env("MINIMUM_REWARD", -300.0)
MAXIMUM_REWARD = get_float_env("MAXIMUM_REWARD", 300.0)

class ModelConfig:
    # AI RELATED VALUES START HERE
    #### MODEL SET UP ####
    HIDDEN_STATE_SIZE = get_int_env("HIDDEN_STATE_SIZE", 1024)
    NUM_RNN_CELLS = get_int_env("NUM_RNN_CELLS", 2)
    LSTM_SIZE = int(HIDDEN_STATE_SIZE / (NUM_RNN_CELLS * 2))
    RNN_SIZES = [LSTM_SIZE] * NUM_RNN_CELLS
    LAYER_HIDDEN_SIZE = get_int_env("LAYER_HIDDEN_SIZE", 1024)
    ROOT_DIRICHLET_ALPHA = get_float_env("ROOT_DIRICHLET_ALPHA", 1.0)
    ROOT_EXPLORATION_FRACTION = get_float_env("ROOT_EXPLORATION_FRACTION", 0.25)
    VISIT_TEMPERATURE = get_float_env("VISIT_TEMPERATURE", 1.0)

    PB_C_BASE = get_int_env("PB_C_BASE", 19652)
    PB_C_INIT = get_float_env("PB_C_INIT", 1.25)

    # ACTION_DIM = 10
    ENCODER_NUM_STEPS = get_int_env("ENCODER_NUM_STEPS", 601)
    SELECTED_SAMPLES = environ.get("SELECTED_SAMPLES", True)
    if isinstance(SELECTED_SAMPLES, str):
        SELECTED_SAMPLES = eval(SELECTED_SAMPLES)
    MAX_GRAD_NORM = get_int_env("MAX_GRAD_NORM", 5)

    N_HEAD_HIDDEN_LAYERS = 2
    N_HEADS = 4
    N_LAYERS = 4
    NUM_SAMPLES = get_int_env("NUM_SAMPLES", 30)
    NUM_SIMULATIONS = get_int_env("NUM_SIMULATIONS", 50)

    ITEM_EMBEDDING_DIM = get_int_env("ITEM_EMBEDDING_DIM", 60)
    CHAMPION_EMBEDDING_DIM = get_int_env("CHAMPION_EMBEDDING_DIM", 512)
    SHOP_EMBEDDING_DIM = get_int_env("SHOP_EMBEDDING_DIM", 64)

class PPOConfig:
    EXP_NAME = environ.get("PPO_EXP_NAME", path.basename(__file__).rstrip(".py"))
    LEARNING_RATE = get_float_env("PPO_LEARNING_RATE", 2.5e-3)
    NUM_ENVS = get_int_env("PPO_NUM_ENVS", 32)
    NUM_STEPS = get_int_env("PPO_NUM_STEPS", 16)
    ANNEAL_LR = get_bool_env("PPO_ANNEAL_LR", "True")
    TOTAL_TIMESTEPS = get_int_env("PPO_TOTAL_TIME_STEPS", 1000000)
    GAE = get_bool_env("PPO_GAE", "True")
    GAMMA = get_float_env("PPO_GAMMA", 0.99)
    GAE_LAMBDA = get_float_env("PPO_GAE_LAMBDA", 0.95)
    NUM_MINIBATCHES = get_int_env("PPO_NUM_MINIBATCHES", 4)
    BATCH_SIZE = int(NUM_ENVS * NUM_STEPS)
    MINIBATCH_SIZE = int(BATCH_SIZE // NUM_MINIBATCHES)
    UPDATE_EPOCHS = get_int_env("PPO_UPDATE_EPOCHS", 1)
    CLIP_COEF = get_float_env("PPO_CLIP_COEF", 0.2)
    CLIP_VLOSS = get_bool_env("PPO_CLIP_VLOSS", "False")
    ENT_COEF = get_float_env("PPO_ENT_COEF", 0.02)
    VF_COEF = get_float_env("PPO_VF_COEF", 1)
    KL_COEF = get_float_env("PPO_KL_COEF", 0.3)
    MAX_GRAD_NORM = get_float_env("PPO_MAX_GRAD_NORM", 0.5)
    TARGET_KL = 0.05
    KL_ADJUSTER = 0.5
    MAX_KL_COEF = 1
    MIN_KL_COEF = 0.01


