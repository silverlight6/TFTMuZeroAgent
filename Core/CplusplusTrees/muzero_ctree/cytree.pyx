# distutils: language=c++
import ctypes
cimport cython
from ctree cimport CMinMaxStatsList, CNode, CRoots, CSearchResults, cbatch_back_propagate, cbatch_traverse
from libcpp.vector cimport vector
from libcpp.list cimport list as cpplist
from libc.stdlib cimport malloc, free

import numpy as np
cimport numpy as np

ctypedef np.npy_float FLOAT
ctypedef np.npy_intp INTP


cdef class MinMaxStatsList:
    cdef CMinMaxStatsList *cmin_max_stats_lst

    def __cinit__(self, int num):
        self.cmin_max_stats_lst = new CMinMaxStatsList(num)

    # def set_delta(self, float value_delta_max):
    #     for i in range(self.cmin_max_stats_lst.num):
    #         self.cmin_max_stats_lst[i].set_delta(value_delta_max)

    def set_delta(self, float value_delta_max):
        self.cmin_max_stats_lst[0].set_delta(value_delta_max)
        
    def __dealloc__(self):
        del self.cmin_max_stats_lst


cdef class ResultsWrapper:
    cdef CSearchResults cresults

    def __cinit__(self, int num):
        self.cresults = CSearchResults(num)

    def get_search_len(self):
        return self.cresults.search_lens


cdef class Roots:
    cdef int root_num
    cdef CRoots *roots

    # root_num is batch_size, tree_nodes is simulation_count, and max_size is space per node
    def __cinit__(self, int root_num, int tree_nodes, max_size):
        self.root_num = root_num
        self.roots = new CRoots(root_num, max_size * (tree_nodes + 2))

    def prepare(self, float root_exploration_fraction, list noises, list reward_pool, list policy_logits_pool,
                list action_nums, list action_counts, list action_limits):
        self.roots[0].prepare(root_exploration_fraction, noises, reward_pool, policy_logits_pool, action_nums,
                              action_counts, action_limits)

    def prepare_no_noise(self, list reward_pool, list policy_logits_pool, list action_nums, list action_counts,
                         list action_limits):
        self.roots[0].prepare_no_noise(reward_pool, policy_logits_pool, action_nums, action_counts, action_limits)

    def get_distributions(self):
        return self.roots[0].get_distributions()

    def get_values(self):
        return self.roots[0].get_values()

    def __dealloc__(self):
        del self.roots

    @property
    def num(self):
        return self.root_num


def batch_back_propagate(int hidden_state_index_x, float discount, list rewards, list values, list policy,
                         MinMaxStatsList min_max_stats_lst, ResultsWrapper results, list action_nums, list action_counts,
                         list action_limits):
    cbatch_back_propagate(hidden_state_index_x, discount, rewards, values, policy,
                          min_max_stats_lst.cmin_max_stats_lst, results.cresults, action_nums, action_counts,
                          action_limits)


def batch_traverse(Roots roots, int pb_c_base, float pb_c_init, float discount, MinMaxStatsList min_max_stats_lst,
                   ResultsWrapper results):

    cbatch_traverse(roots.roots, pb_c_base, pb_c_init, discount, min_max_stats_lst.cmin_max_stats_lst, results.cresults)

    return results.cresults.hidden_state_index_x_lst, results.cresults.hidden_state_index_y_lst, results.cresults.last_actions, results.cresults.search_lens
