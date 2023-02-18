import numpy as np
from functools import wraps
from time import time


def champ_binary_encode(n):
    return list(np.unpackbits(np.array([n], np.uint8))[2:8])


def champ_binary_decode(array):
    temp = list(array.copy().astype(int))
    temp.insert(0, 0)
    temp.insert(0, 0)
    return np.packbits(temp, axis=-1)[0]


def item_binary_encode(n):
    return list(np.unpackbits(np.array([n], np.uint8))[2:8])


def champ_one_hot_encode(n):
    MAX_CHAMPION_IN_SET = 58
    return np.eye(MAX_CHAMPION_IN_SET)[n]


def item_one_hot_encode(n):
    return np.eye(9)[n]


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


def decode_action(str_actions):
    actions = []
    for str_action in str_actions:
        num_items = str_action.count("_")
        split_action = str_action.split("_")
        element_list = [0, 0, 0]
        for i in range(num_items + 1):
            element_list[i] = int(split_action[i])
        actions.append(np.asarray(element_list))
    return np.asarray(actions)