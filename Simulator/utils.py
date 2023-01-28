def champ_binary_encode(self, n):
    return list(np.unpackbits(np.array([n],np.uint8))[2:8])

def item_binary_encode(self, n):
    return list(np.unpackbits(np.array([n],np.uint8))[2:8])

def champ_one_hot_encode(self, n):
    return self.CHAMPION_ONE_HOT_ENCODING[n]

def item_one_hot_encode(self, n):
    return self.BASIC_ITEMS_ONE_HOT_ENCODING[n]