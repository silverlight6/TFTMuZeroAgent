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
TRAINER_GPU_SIZE = get_float_env("TRAINER_GPU_SIZE", 0.2) if DEVICE == "cuda" else 0

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
SCALAR_INPUT_SIZE = 25
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
    # Hidden State Size is the number of neurons in the hidden state. This is the primary variable for model size
    HIDDEN_STATE_SIZE = get_int_env("HIDDEN_STATE_SIZE", 1024)
    # LSTM controls. Any other number than 2 has not been tested with
    NUM_RNN_CELLS = get_int_env("NUM_RNN_CELLS", 2)
    LSTM_SIZE = int(HIDDEN_STATE_SIZE / (NUM_RNN_CELLS * 2))
    RNN_SIZES = [LSTM_SIZE] * NUM_RNN_CELLS
    # Layer Hidden Size -> Inner layer size for the MLP blocks.
    LAYER_HIDDEN_SIZE = get_int_env("LAYER_HIDDEN_SIZE", 1024)
    # Used to create dirichlet noise to allow for exploration in the MCTS tree
    ROOT_DIRICHLET_ALPHA = get_float_env("ROOT_DIRICHLET_ALPHA", 1.0)
    # Base exploration fraction
    ROOT_EXPLORATION_FRACTION = get_float_env("ROOT_EXPLORATION_FRACTION", 0.25)
    # How much to expand or contract the policy function. 1.0 means do not touch.
    VISIT_TEMPERATURE = get_float_env("VISIT_TEMPERATURE", 1.0)

    # Base values from the MuZero paper for base and init
    PB_C_BASE = get_int_env("PB_C_BASE", 19652)
    PB_C_INIT = get_float_env("PB_C_INIT", 1.25)

    # This is the number of neurons on the value and reward network.
    # This maps to a number from -300 to 300. A more detailed explanation can be found in the MuZero paper
    ENCODER_NUM_STEPS = get_int_env("ENCODER_NUM_STEPS", 601)
    # Maximum norm for the training gradient
    MAX_GRAD_NORM = get_int_env("MAX_GRAD_NORM", 5)

    # Used in a few experiments but need cleaning up
    N_HEAD_HIDDEN_LAYERS = 2
    N_HEADS = 4
    N_LAYERS = 4
    # Number of sampled actions to pull from the total number of actions
    NUM_SAMPLES = get_int_env("NUM_SAMPLES", 30)
    # Number of MCTS Simulations to expand on every action
    NUM_SIMULATIONS = get_int_env("NUM_SIMULATIONS", 50)

    # Size of champion embedding
    CHAMPION_EMBEDDING_DIM = get_int_env("CHAMPION_EMBEDDING_DIM", 512)
    # Size of the shop embeddings
    SHOP_EMBEDDING_DIM = get_int_env("SHOP_EMBEDDING_DIM", 64)

# Commenting in the code for PPO Config.
class PPOConfig:
    # Name of the experiment. This will be used for save files. Fill in the details on the default later
    EXP_NAME = environ.get("PPO_EXP_NAME", path.basename(__file__).rstrip(".py"))
    # This is the learning rate specifically for PPO.
    # This is separate from the MuZero section because there are times when using a different learning rate than
    # MuZero. During the training of the Guide Model, both models are training at the same time so having separate
    # Learning rates is useful.
    LEARNING_RATE = get_float_env("PPO_LEARNING_RATE", 2.5e-4)
    # Number of separate environments to be loaded in parallel
    NUM_ENVS = get_int_env("PPO_NUM_ENVS", 64)
    # Number of steps that each of those environments takes before returning their data
    NUM_STEPS = get_int_env("PPO_NUM_STEPS", 16)
    # If we want to lower the learning over the course of training. A simple linear function is currently used for
    # learning rate anneal.
    ANNEAL_LR = get_bool_env("PPO_ANNEAL_LR", "True")
    # How long do we want to train our model for. When checkpoints get implemented, this is going to be traded for a
    # while(True) loop.
    TOTAL_TIMESTEPS = get_float_env("PPO_TOTAL_TIME_STEPS", 10000000)
    # Will mention here that GAMMA, GAE, and LAMBDA are elimated due to this being a single step implementation of PPO
    # How many minibatches to run before combining into a full batch and updating the weights of the model.
    NUM_MINIBATCHES = get_int_env("PPO_NUM_MINIBATCHES", 4)
    # Full batch size.
    BATCH_SIZE = int(NUM_ENVS * NUM_STEPS)
    # Mini batch size.
    MINIBATCH_SIZE = int(BATCH_SIZE // NUM_MINIBATCHES)
    # How many times do we want to use the data we gathered to update the model.
    # Since the Position and Item Simulator are very fast and collecting a full batch can be done in around 2 seconds,
    # The number of updates is currently set to 1. If collecting data from the simulator was the main form of
    # computational restraint, then setting this number higher would be beneficial.
    UPDATE_EPOCHS = get_int_env("PPO_UPDATE_EPOCHS", 1)
    # Factor on how much to clip the policy loss. 0.2 is standard
    CLIP_COEF = get_float_env("PPO_CLIP_COEF", 0.2)
    # Boolean on clipping the value loss. Set to false currently to allow large changes early in the policy and value
    # functions
    CLIP_VLOSS = get_bool_env("PPO_CLIP_VLOSS", "False")
    # Entropy coefficient. Set low to keep the entropy loss less than that of the poilcy and value
    ENT_COEF = get_float_env("PPO_ENT_COEF", 0.005)
    # Value Function coefficient. Set at 0.5 to keep in line with the policy loss
    VF_COEF = get_float_env("PPO_VF_COEF", 0.5)
    # KL_Loss Coefficient. Currently, experimenting with this but when the KL stabilizes around 0.1, keeping this
    # around 0.5 makes it in line with the value and policy loss. In general, large changes to the policy and value
    # network are generally acceptable early in training but have yet to test the results as training gets deeper
    KL_COEF = get_float_env("PPO_KL_COEF", 0.01)
    # Make sure the gradient stays within reason to prevent leaving the function space entirely
    MAX_GRAD_NORM = get_float_env("PPO_MAX_GRAD_NORM", 0.5)
    # What we want the KL to hover around
    TARGET_KL = 0.1
    # How much to change the KL_COEF if the target kl is either too high or too low.
    KL_ADJUSTER = 1.5


