import Simulator.config as config
import numpy as np
from functools import wraps
from time import time
from Simulator.item_stats import item_builds, uncraftable_items

from Simulator.stats import COST


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
    return np.eye(config.MAX_CHAMPION_IN_SET)[n]


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
        element_list = []
        for i in range(num_items + 1):
            element_list.append(int(split_action[i]))
        while len(element_list) < 3:
            element_list.append(0)
        actions.append(np.asarray(element_list))
    return np.asarray(actions)


def x_y_to_1d_coord(x1, y1):
    if y1 == -1:
        return x1 + 28
    else:
        return 7 * y1 + x1


def null_encode(champ_object):
    return None


def encode_champ_object(champ_object):
    # when using binary encoding (58 champ + stars + chosen) = 60
    champion_info_array = np.zeros((1 * 58 + 1 + 1, 1))
    if champ_object is None or COST[champ_object.name] == 0:
        return champion_info_array
    c_index = list(COST.keys()).index(champ_object.name)
    champion_info_array[0:58, 0] = champ_one_hot_encode(c_index - 1)
    champion_info_array[58, 0] = champ_object.stars
    champion_info_array[59, 0] = 1 if champ_object.chosen else 0
    # for ind, item in enumerate(champ_object.items):
    #     start = (ind * 6) + 7
    #     finish = start + 6
    #     i_index = []
    #     if item in uncraftable_items:
    #         i_index = list(uncraftable_items).index(item) + 1
    #     elif item in item_builds.keys():
    #         i_index = list(item_builds.keys()).index(item) + 1 + len(uncraftable_items)
    #     champion_info_array[start:finish, 0] = utils.item_binary_encode(i_index)
    return champion_info_array


def encode_item_object(item_object):
    item_info = np.zeros((60, 1))
    if item_object in uncraftable_items:
        item_info[0:6, 0] = item_binary_encode(list(uncraftable_items).index(item_object) + 1)
    elif item_object in item_builds.keys():
        item_info[0:6, 0] = item_binary_encode(
            list(item_builds.keys()).index(item_object) + 1 + len(uncraftable_items))
    return item_info


class Encoded_List:
    def __init__(self, len, encoding) -> None:
        self._list = [None for _ in range(len)]
        self._encoded_list = np.zeros([60, 1, len])
        self.encoding = encoding

    def __getitem__(self, key):
        return self._list[key]

    def __setitem__(self, key, value):
        self._list[key] = value
        self._encoded_list[:, :, key] = self.encoding(self._list[key])

    def __len__(self):
        return len(self._list)

    def get_encoding(self):
        return self._encoded_list
