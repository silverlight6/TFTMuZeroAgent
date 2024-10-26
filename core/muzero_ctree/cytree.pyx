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

    def get_max(self, int num):
        return self.cmin_max_stats_lst[0].get_max(num)

    def get_min(self, int num):
        return self.cmin_max_stats_lst[0].get_min(num)
        
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
    cdef int pool_size
    cdef CRoots *roots

    def __cinit__(self, int root_num, int tree_nodes, max_size):
        self.root_num = root_num
        self.pool_size = max_size * (tree_nodes + 2)
        self.roots = new CRoots(root_num, self.pool_size)

    def prepare(self, float root_exploration_fraction, list noises, list reward_pool, list policy_logits_pool,
                list action_nums, list action_counts):
        self.roots[0].prepare(root_exploration_fraction, noises, reward_pool, policy_logits_pool, action_nums, action_counts)

    def prepare_no_noise(self, list reward_pool, list policy_logits_pool, list action_nums, list action_counts):
        self.roots[0].prepare_no_noise(reward_pool, policy_logits_pool, action_nums, action_counts)

    def get_distributions(self):
        return self.roots[0].get_distributions()

    def get_values(self):
        return self.roots[0].get_values()

    def __dealloc__(self):
        del self.roots

    @property
    def num(self):
        return self.root_num


cdef class Node:
    cdef CNode cnode

    def __cinit__(self):
        pass

    def __cinit__(self, float prior, int action_num):
        # self.cnode = CNode(prior, action_num)
        pass

    def expand(self, int hidden_state_index_x, int hidden_state_index_y, float value_prefix,
               list policy_logits, int act_num, int action_count):
        cdef vector[float] cpolicy = policy_logits
        self.cnode.expand(hidden_state_index_x, hidden_state_index_y, value_prefix, cpolicy, act_num, action_count)

def batch_back_propagate(int hidden_state_index_x, float discount, list rewards, list values, list policy,
                         MinMaxStatsList min_max_stats_lst, ResultsWrapper results, list action_nums, list action_counts):
    cdef int i
    cdef vector[float] crewards = rewards
    cdef vector[float] cvalues = values
    cdef vector[vector[float]] cpolicy = policy
    cdef vector[int] caction_nums = action_nums
    cdef vector[int] caction_counts = action_counts

    cbatch_back_propagate(hidden_state_index_x, discount, crewards, cvalues, cpolicy,
                          min_max_stats_lst.cmin_max_stats_lst, results.cresults, caction_nums, caction_counts)


def batch_traverse(Roots roots, int pb_c_base, float pb_c_init, float discount, MinMaxStatsList min_max_stats_lst,
                   ResultsWrapper results):

    cbatch_traverse(roots.roots, pb_c_base, pb_c_init, discount, min_max_stats_lst.cmin_max_stats_lst, results.cresults)

    return results.cresults.hidden_state_index_x_lst, results.cresults.hidden_state_index_y_lst, results.cresults.last_actions, results.cresults.search_lens
