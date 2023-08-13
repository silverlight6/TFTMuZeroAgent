import numpy as np

# IMPORTANT: Change this value to the number of cpu cores you want to use (recommended 80% of cpu)
NUM_CPUS = 26
GPU_SIZE_PER_WORKER = 0.15
STORAGE_GPU_SIZE = 0.15

DEVICE = "cuda"
STOCHASTIC = True

# AI RELATED VALUES START HERE

#### MODEL SET UP ####
HIDDEN_STATE_SIZE = 1024
NUM_RNN_CELLS = 4
LSTM_SIZE = int(HIDDEN_STATE_SIZE / (NUM_RNN_CELLS * 2))
RNN_SIZES = [LSTM_SIZE] * NUM_RNN_CELLS
LAYER_HIDDEN_SIZE = 512
ROOT_DIRICHLET_ALPHA = 0.5
ROOT_EXPLORATION_FRACTION = 0.25
MINIMUM_REWARD = -300.0
MAXIMUM_REWARD = 300.0
PB_C_BASE = 19652
PB_C_INIT = 1.25
DISCOUNT = 0.997
TRAINING_STEPS = 1e10
OBSERVATION_SIZE = 7806
OBSERVATION_TIME_STEPS = 2
OBSERVATION_TIME_STEP_INTERVAL = 4
INPUT_TENSOR_SHAPE = np.array([OBSERVATION_SIZE])
ACTION_ENCODING_SIZE = 1045
ACTION_CONCAT_SIZE = 81
ACTION_DIM = [7, 37, 10]


POLICY_HEAD_SIZES = [7, 5, 630, 370, 9]  # [7 types, shop, movement, item, sell/item loc]
NEEDS_2ND_DIM = [1, 2, 3, 4]

# ACTION_DIM = 10
ENCODER_NUM_STEPS = 601
SELECTED_SAMPLES = True
MAX_GRAD_NORM = 5

N_HEAD_HIDDEN_LAYERS = 2

### TIME RELATED VALUES ###
ACTIONS_PER_TURN = 15
CONCURRENT_GAMES = 20
NUM_PLAYERS = 8 
NUM_SAMPLES = 25
NUM_SIMULATIONS = 50

# Set to -1 to turn off.
TD_STEPS = -1 
# This should be 1000 + because we want to be sampling everything when using priority. To change, look into the code in replay_muzero_buffer
SAMPLES_PER_PLAYER = 1000  
UNROLL_STEPS = 5

### TRAINING ###
BATCH_SIZE = 12
INIT_LEARNING_RATE = 0.01
LEARNING_RATE_DECAY = int(350e3)
LR_DECAY_FUNCTION = 0.1
WEIGHT_DECAY = 1e-5
REWARD_LOSS_SCALING = 0
POLICY_LOSS_SCALING = 1

AUTO_BATTLER_PERCENTAGE = 1

# Putting this here so that we don't scale the policy by a multiple of 5
# Because we calculate the loss for each of the 5 dimensions.
# I'll add a mathematical way of generating these numbers later.
DEBUG = True
CHECKPOINT_STEPS = 100

#### TESTING ####
RUN_UNIT_TEST = False
RUN_PLAYER_TESTS = False
RUN_MINION_TESTS = False
RUN_DROP_TESTS = False
RUN_MCTS_TESTS = False
RUN_MAPPING_TESTS = False
LOG_COMBAT = False
