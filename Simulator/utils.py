import numpy as np

def champ_binary_encode(n):
    return list(np.unpackbits(np.array([n],np.uint8))[2:8])

def item_binary_encode(n):
    return list(np.unpackbits(np.array([n],np.uint8))[2:8])

def champ_one_hot_encode(n):
    return np.eye(58)[n] #58 = Max champion in set

def item_one_hot_encode(n):
    return np.eye(9)[n] #Amount of basic items a champ can hold in set

def one_hot_encode_number(number, depth):
    return np.eye(depth)[number]