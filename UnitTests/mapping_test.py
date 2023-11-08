from Models import MCTS_Util as utils

def setup():
    default_mapping = utils.create_default_mapping()
    return default_mapping

def mapping_test():
    default_mapping = setup()

    for dim in range(len(default_mapping)):
        local_mapping = default_mapping[dim][0]

        if dim == 0:  # type dim; 7; "0", "1", ... "6"
            for type_action in local_mapping:
                mapped_idx = int(type_action)
                assert type_action == local_mapping[mapped_idx]

        elif dim == 1: # shop dim; 5; "_0", "_1", ... "_4"
            for shop_action in local_mapping:
                mapped_idx = int(shop_action[1])  # "_1" -> "1"
                assert shop_action == local_mapping[mapped_idx]

        elif dim == 2: # board dim; 630; "_0_1", "_0_2", ... "_37_28"
            for board_action in local_mapping:
                board_action = board_action.split('_')  # "_20_21" -> ["", "20", "21"]
                from_loc = int(board_action[1])  # ["", "20", "21"] -> "20"
                to_loc = int(board_action[2])  # ["", "20", "21"] -> "21"
                mapped_idx = sum([35 - i for i in range(from_loc)]) + (to_loc - 1)
                assert board_action == local_mapping[mapped_idx].split('_')

        elif dim == 3: # item dim; 370; "_0_0", "_0_1", ... "_9_36"
            for item_action in local_mapping:
                item_action = item_action.split('_')  # "_10_9" -> ["", "10", "9"]
                item_loc = int(item_action[1])  # "_0_20" -> "0"
                champ_loc = int(item_action[2])  # "_0_20" -> "20"
                mapped_idx = (10 * item_loc) + (champ_loc)
                assert item_action == local_mapping[mapped_idx].split('_')

        elif dim == 4:  # sell dim; 37; "_0", "_1", "_36"
            for sell_action in local_mapping:
                mapped_idx = int(sell_action[1:])  # "_15" -> "15"
                assert sell_action == local_mapping[mapped_idx]

def test_list():
    mapping_test()
