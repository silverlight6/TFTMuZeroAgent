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

namespace tree {

    class CNode {
        public:
            int visit_count, action_num, hidden_state_index_x, hidden_state_index_y, terminal_children;
            float reward, prior, value_sum;
            bool terminal;
            std::vector<int> children_index;
            std::vector<CNode>* ptr_node_pool;

            CNode();
            CNode(float prior, std::vector<CNode> *ptr_node_pool, bool terminal);
            ~CNode();

            void expand(int hidden_state_index_x, int hidden_state_index_y, float reward,
                        const std::vector<float> &policy_logits, int act_num, int action_count, int action_limit);
            void add_exploration_noise(float exploration_fraction, const std::vector<float> &noises);
            float compute_mean_q(int isRoot, float parent_q, float discount_factor);


            int expanded();

            float value();
            float qvalue(float discount);

            std::vector<int> get_children_distribution();
            CNode* get_child(int action);
    };

    class CRoots{
        public:
            int root_num, pool_size;
            std::vector<CNode> roots;
            std::vector<std::vector<CNode>> node_pools;

            CRoots();
            CRoots(int root_num, int pool_size);
            ~CRoots();

            void prepare(float root_exploration_fraction, const std::vector<std::vector<float>> &noises,
                         const std::vector<float> &rewards, const std::vector<std::vector<float>> &policies,
                         const std::vector<int> &action_nums, const std::vector<int> &action_counts,
                         const std::vector<int> &action_limits);
            void prepare_no_noise(const std::vector<float> &rewards, const std::vector<std::vector<float>> &policies,
                                  const std::vector<int> &action_nums, const std::vector<int> &action_counts,
                                  const std::vector<int> &action_limits);
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
    void cback_propagate(std::vector<CNode*> &search_path, tools::CMinMaxStats &min_max_stats, float value,
                         float discount);
    void cupdate_visit_count(std::vector<CNode*> &search_path);
    void cbatch_back_propagate(int hidden_state_index_x, float discount, const std::vector<float> &rewards,
                               const std::vector<float> &values, const std::vector<std::vector<float>> &policy,
                               tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results,
                               const std::vector<int> &action_nums, const std::vector<int> &action_counts,
                               const std::vector<int> &action_limits);
    int cselect_child(CNode* root, tools::CMinMaxStats &min_max_stats, int pb_c_base, float pb_c_init, float discount,
                      float mean_q);
    float cucb_score(CNode *child, tools::CMinMaxStats &min_max_stats, float parent_mean_q,
                     float total_children_visit_counts, float pb_c_base, float pb_c_init, float discount);
    void cbatch_traverse(CRoots *roots, int pb_c_base, float pb_c_init, float discount,
                         tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results);
}

#endif