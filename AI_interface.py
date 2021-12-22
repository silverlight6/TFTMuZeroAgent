import champion
import player as player_class
from pool import pool
from champion_functions import MILLIS
from stats import COST


# This will be the AI interface. Sampling from an LSTM
def ai_interface(self, player):
    num_iterations = 20000 # @param {type:"integer"}

    initial_collect_steps = 100  # @param {type:"integer"}
    collect_steps_per_iteration =   1# @param {type:"integer"}
    replay_buffer_max_length = 100000  # @param {type:"integer"}

    batch_size = 16  # @param {type:"integer"}
    learning_rate = 1e-3  # @param {type:"number"}
    log_interval = 200  # @param {type:"integer"}

    num_eval_episodes = 10  # @param {type:"integer"}
    eval_interval = 1000  # @param {type:"integer"}
    return

def generate_shop_vector(self, shop):
    # each champion has 6 bit for the name, 1 bit for the chosen.
    # 5 of them makes it 35.
    output_array = np.zeros(35)
    for x in range(0, len(shop)):
        input_array = np.zeros(7)
        if shop[x]:
            i_index = list(COST.keys()).index(shop[x])
            # This should update the item name section of the vector
            for z in range(0, 6, -1):
                if i_index > 2 * z:
                    input_array[6 - z] = 1
                    i_index -= 2 * z 
        # Input chosen mechanics once I go back and update the chosen mechanics. 
        output_array(7 * x: 7 * x + 1) = input_array
    return output_array