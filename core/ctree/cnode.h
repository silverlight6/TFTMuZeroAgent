#ifndef CNODE_H
#define CNODE_H

#include <vector>
#include <stack>
#include <string>
#include <cmath>
#include <algorithm>
#include "cminimax.h"
#include <stdlib.h>
#include <math.h>

const int DEBUG_MODE = 0;

namespace tree {

    class CNode {
        public:
            int visit_count, to_play, action_num, hidden_state_index_x, hidden_state_index_y, best_action, is_reset;
            float value_prefix, prior, value_sum;
            std::vector<int> children_index;
            std::vector<CNode>* ptr_node_pool;
            // This is used to map the action from the 1d to the multi dim that the environment can use
            // I wish this could be a string vector but Cython does not support strings, only char *
            std::vector<char*> mappings;

            CNode();
            CNode(float prior, int action_num, std::vector<CNode> *ptr_node_pool);
            ~CNode();

            void expand(int to_play, int hidden_state_index_x, int hidden_state_index_y, float value_prefix,
                        const std::vector<float> &policy_logits, const std::vector<char*> mappings);
            void add_exploration_noise(float exploration_fraction, const std::vector<float> &noises);
            float get_mean_q(int isRoot, float parent_q, float discount);

            int expanded();

            float value();

            std::vector<int> get_trajectory();
            std::vector<int> get_children_distribution();
            CNode* get_child(int action);
    };

    class CRoots{
        public:
            int root_num, pool_size;
            int action_num;
            std::vector<CNode> roots;
            std::vector<std::vector<CNode>> node_pools;

            CRoots();
            CRoots(int root_num, int action_num, int pool_size);
            ~CRoots();

            void prepare(float root_exploration_fraction, const std::vector<std::vector<float>> &noises,
                         const std::vector<float> &value_prefixs, const std::vector<std::vector<float>> &policies,
                         const std::vector<std::vector<char*>> &mappings);
            void prepare_no_noise(const std::vector<float> &value_prefixs,
                                  const std::vector<std::vector<float>> &policies,
                                  const std::vector<std::vector<char*>> &mappings);
            void clear();
            std::vector<std::vector<int>> get_trajectories();
            std::vector<std::vector<int>> get_distributions();
            std::vector<float> get_values();

    };

    class CSearchResults{
        public:
            int num;
            std::vector<int> hidden_state_index_x_lst, hidden_state_index_y_lst, search_lens;
            std::vector<std::vector<int>> last_actions;
            std::vector<CNode*> nodes;
            std::vector<std::vector<CNode*>> search_paths;

            CSearchResults();
            CSearchResults(int num);
            ~CSearchResults();

    };


    //*********************************************************
    void update_tree_q(CNode* root, tools::CMinMaxStats &min_max_stats, float discount);
    std::vector<int> decode_action(char* &str_action);
    std::vector<char*> create_default_mapping();
    void cback_propagate(std::vector<CNode*> &search_path, tools::CMinMaxStats &min_max_stats, int to_play,
                         float value, float discount);
    void cbatch_back_propagate(int hidden_state_index_x, float discount, const std::vector<float> &value_prefixs,
                               const std::vector<float> &values, const std::vector<std::vector<float>> &policy,
                               tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results,
                               std::vector<int> is_reset_lst, std::vector<std::vector<char*>> mappings);
    int cselect_child(CNode* root, tools::CMinMaxStats &min_max_stats, int pb_c_base, float pb_c_init,
                      float discount, float mean_q);
    float cucb_score(CNode *child, tools::CMinMaxStats &min_max_stats, float parent_mean_q, int is_reset,
                     float total_children_visit_counts, float parent_value_prefix, float pb_c_base,
                     float pb_c_init, float discount);
    void cbatch_traverse(CRoots *roots, int pb_c_base, float pb_c_init, float discount,
                         tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results);
}

#endif