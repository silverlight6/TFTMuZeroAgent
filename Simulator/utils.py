import numpy as np
from functools import wraps
from time import time
import config

def champ_binary_encode(n):
    return list(np.unpackbits(np.array([n],np.uint8))[2:8])

def champ_binary_decode(array):
    temp = list(array.copy().astype(int))
    temp.insert(0,0)
    temp.insert(0,0)
    return np.packbits(temp, axis=-1)[0]

def item_binary_encode(n):
    return list(np.unpackbits(np.array([n],np.uint8))[2:8])

def champ_one_hot_encode(n):
    return np.eye(58)[n] #58 = Max champion in set

def item_one_hot_encode(n):
    return np.eye(9)[n] #Amount of basic items a champ can hold in set

def one_hot_encode_number(number, depth):
    return np.eye(depth)[number]

def timed(f):
  @wraps(f)
  def wrapper(*args, **kwds):
    start = time()
    result = f(*args, **kwds)
    elapsed = time() - start
    print(f'{f.__name__} took {elapsed} seconds to finish')
    return result
  return wrapper

def generate_masking(action):
        num_items = action.count("_")
        split_action = action.split("_")
        element_list = [0,0,0]
        for i in range(num_items+1):
            element_list[i] = int(split_action[i])
        
        mask = np.zeros(config.ACTION_DIM[0] + config.ACTION_DIM[1] + config.ACTION_DIM[2])

        mask[0:6] = np.ones(6)
        if element_list[0] == 1:
            mask[6:11] = np.ones(5)
        elif element_list[0] == 2:
            mask[6:44] = np.ones(38)
        elif element_list[0] == 3:
            mask = np.ones(54)
        return mask