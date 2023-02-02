import numpy as np

def champ_binary_encode(n):
    return list(np.unpackbits(np.array([n],np.uint8))[2:8])

def item_binary_encode(n):
    return list(np.unpackbits(np.array([n],np.uint8))[2:8])

def champ_one_hot_encode(n):
    MAX_CHAMPION_IN_SET = 58
    return np.eye(MAX_CHAMPION_IN_SET)[n]

def item_one_hot_encode(n):
    return np.eye(9)[n]