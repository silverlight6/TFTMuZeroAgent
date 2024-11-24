# Import required components
import core.muzero_ctree.cytree as tree

def create_roots():
    # Initialize a Roots object for testing
    return tree.Roots(2, 10, 30)

def create_minmax_stats_list(num):
    # Sample MinMaxStatsList setup
    return tree.MinMaxStatsList(num)

# -----------------------------------------
# Test for CRoots class
# -----------------------------------------

def roots_initialization_test(roots):
    assert roots.num == 2

def roots_prepare_test(roots):
    roots.prepare(0.25, [[0.2, 0.3], [0.1, 0.4]], [1.0, 2.0],
                  [[0.1, 0.9], [0.3, 0.7]], [1, 2], [3, 4], [5, 6])
    dist = roots.get_distributions()
    assert isinstance(dist, list)
    assert len(dist) == 2

def roots_prepare_no_noise_test(roots):
    roots.prepare_no_noise([1.0, 2.0], [[0.5, 0.5], [0.3, 0.7]],
                           [2, 2], [2, 2], [10, 10])
    dist = roots.get_distributions()
    assert isinstance(dist, list)
    assert len(dist) == 2

# -----------------------------------------
# Test for batch_back_propagate and batch_traverse
# -----------------------------------------

def batch_back_propagate_test(roots):
    # Sample input data for the batch backpropagation
    num = roots.num
    minmax_stats_list = create_minmax_stats_list(num)
    results = tree.ResultsWrapper(num)
    hidden_state_index_x_lst, hidden_state_index_y_lst, last_action, search_lens = \
        tree.batch_traverse(roots, 19652, 1.25, 0.98, minmax_stats_list, results)
    tree.batch_back_propagate(0, 0.9, [1.0, 2.0], [3.0, 4.0], [[0.2, 0.8], [0.4, 0.6]],
                              minmax_stats_list, results, [2, 2], [2, 2], [10, 10])
    assert hidden_state_index_x_lst == [0, 0]
    assert hidden_state_index_y_lst == [0, 1]
    assert search_lens == [1, 1]
    assert len(results.get_search_len()) == 2

# I think tomorrow I need to think about adding a stress test to this

def test_list():
    roots = create_roots()
    roots_initialization_test(roots)
    roots_prepare_test(roots)
    roots_prepare_no_noise_test(roots)
    batch_back_propagate_test(roots)
