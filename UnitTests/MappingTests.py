from Models import MCTS_Util as utils

def setup():
    _, default_mapping = utils.create_default_mapping()
    default_mapping = default_mapping[0]
    return default_mapping

def mapping_test():
    default_mapping = setup()
    for action in default_mapping:
        index = utils.flatten_action(action)
        assert default_mapping[index] == action
        
def list_of_tests():
    mapping_test()