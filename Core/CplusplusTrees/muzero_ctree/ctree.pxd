# distutils: language=c++
from libcpp.vector cimport vector

cdef extern from "cminimax.cpp":
    pass

cdef extern from "cminimax.h" namespace "tools":
    cdef cppclass CMinMaxStats:
        CMinMaxStats() except +
        float maximum, minimum

        void set_delta(float value_delta_max)
        void update(float value)
        void clear()
        float normalize(float value)

    cdef cppclass CMinMaxStatsList:
        CMinMaxStatsList() except +
        CMinMaxStatsList(int num) except +
        int num
        vector[CMinMaxStats] stats_lst
        void set_delta(float value_delta_max)

cdef extern from "cnode.cpp":
    pass

cdef extern from "cnode.h" namespace "tree":
    cdef cppclass CNode:
        CNode() except +
        CNode(float prior, vector[CNode]* ptr_node_pool) except +
        int visit_count, action_num, hidden_state_index_x, hidden_state_index_y, terminal_children
        float reward, prior, value_sum
        vector[int] children_index
        vector[CNode]* ptr_node_pool
        bint terminal

        void expand(int hidden_state_index_x, int hidden_state_index_y, float reward,
                    vector[float] policy_logits, int act_num, int action_count, int action_limit)
        void add_exploration_noise(float exploration_fraction, vector[float] noises)

        int expanded()
        float value()
        float qvalue(float discount)
        vector[int] get_children_distribution()
        CNode* get_child(int action)

    cdef cppclass CRoots:
        CRoots() except +
        CRoots(int root_num, int pool_size) except +
        int root_num, pool_size
        vector[CNode] roots
        vector[vector[CNode]] node_pools

        void prepare(float root_exploration_fraction, const vector[vector[float]] &noises,
                     const vector[float] &rewards, const vector[vector[float]] &policies,
                     vector[int] action_nums, vector[int] action_counts, vector[int] action_limits)
        void prepare_no_noise(const vector[float] &rewards, const vector[vector[float]] &policies,
                              vector[int] action_nums, vector[int] action_counts, vector[int] action_limits)
        vector[vector[int]] get_distributions()
        vector[float] get_values()

    cdef cppclass CSearchResults:
        CSearchResults() except +
        CSearchResults(int num) except +
        int num
        vector[int] hidden_state_index_x_lst, hidden_state_index_y_lst, search_lens
        vector[vector[int]] last_actions
        vector[CNode*] nodes

    cdef void cback_propagate(vector[CNode*] &search_path, CMinMaxStats &min_max_stats, float value, float discount)
    void cbatch_back_propagate(int hidden_state_index_x, float discount, vector[float] rewards,
                               vector[float] values, vector[vector[float]] policy, CMinMaxStatsList *min_max_stats_lst,
                               CSearchResults &results, vector[int] action_nums, vector[int] action_counts,
                               vector[int] action_limits)
    void cbatch_traverse(CRoots *roots, int pb_c_base, float pb_c_init, float discount,
                         CMinMaxStatsList *min_max_stats_lst, CSearchResults &results)
