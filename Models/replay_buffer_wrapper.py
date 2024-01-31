import numpy as np
import config
import ray
from Models.replay_muzero_buffer import ReplayBuffer


class BufferWrapper:
    def __init__(self, global_buffer):
        self.buffers = {"player_" + str(i): ReplayBuffer(global_buffer) for i in range(config.NUM_PLAYERS)}

    def reset(self):
        for key in self.buffers.keys():
            self.buffers[key].reset()
    
    def store_replay_buffer(self, key, *args):
        self.buffers[key].store_replay_buffer(args[0], args[1], args[2], args[3], args[4])

    def store_combat_buffer(self, key, combat):
        self.buffers["player_0"].store_combats_buffer(combat)

    def get_prev_action(self, key):
        self.buffers[key].get_prev_action()
    
    def get_reward_sequence(self, key):
        self.buffers[key].get_reward_sequence()
    
    def set_reward_sequence(self, key, *args):
        self.buffers[key].set_reward_sequence(args[0])
    
    def store_global_buffer(self):
        max_lenght = 0
        for b in self.buffers.values():
            max_lenght = max(max_lenght, b.get_len())
        for b in self.buffers.values():
            b.store_global_buffer(max_lenght)
    