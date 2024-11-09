#include <iostream>
#include "cnode.h"

namespace tree {
    CSearchResults::CSearchResults() {
        this->num = 0;
    }

    CSearchResults::CSearchResults(int num) {
        this->num = num;
        for(int i = 0; i < num; ++i){
            this->search_paths.push_back(std::vector<CNode*>());
        }
    }

    CSearchResults::~CSearchResults() {}

    //*********************************************************

    CNode::CNode() {
        this->prior = 0;
        this->action_num = 0;
        this->visit_count = 0;
        this->value_sum = 0;
        this->reward = 0.0;
        this->terminal_children= 0;
        this->terminal = false;
        this->ptr_node_pool = nullptr;
    }

    CNode::CNode(float prior, std::vector<CNode>* ptr_node_pool, bool terminal) {
        this->prior = prior;
        this->action_num = 0;
        this->visit_count = 0;
        this->value_sum = 0;
        this->reward = 0.0;
        this->terminal_children=0;
        this->terminal = terminal;
        this->ptr_node_pool = ptr_node_pool;
        this->hidden_state_index_x = -1;
        this->hidden_state_index_y = -1;
    }

    CNode::~CNode() {}

    // I am not able to find a definite answer to we use the sampled policy as the base for the prior
    // or if we do a weighted softmax which is what is done below.
    // The alpha-go paper uses .67 as their weight but we appear to use e instead.
    void CNode::expand(int hidden_state_index_x, int hidden_state_index_y, float reward,
                       const std::vector<float> &policy_logits, int act_num, int action_count, int action_limit) {
        // Index for finding the hidden state on python side, x is search path location, y is the player
        this->hidden_state_index_x = hidden_state_index_x;
        this->hidden_state_index_y = hidden_state_index_y;
        this->reward = reward;

        // number of unique actions this node contains. Changes based on number of unique samples
        this->action_num = act_num;

        float temp_policy;
        // sum is a float instead of a tensor since we handle 1 player at a time
        float policy_sum = 0.0;
        std::vector<float> policy(action_num);
        float policy_max = FLOAT_MIN;
        // Find the maximum
        for(int a = 0; a < act_num; ++a){
            if(policy_max < policy_logits[a]){
                policy_max = policy_logits[a];
            }
        }

        // Calculate the sum and create a temp policy with the exp of each
        for(int a = 0; a < act_num; ++a) {
            // exp is e ^ value and since all values are negative, all values in temp_policy are between 0 and 1
            temp_policy = exp(policy_logits[a] - policy_max);
            policy_sum += temp_policy;
            policy[a] = temp_policy;
        }

        float prior;
        std::vector<CNode>* ptr_node_pool = this->ptr_node_pool;

        for(int a = 0; a < act_num; ++a) {
            // Normalizes the array
            prior = policy[a] / policy_sum;
            int index = ptr_node_pool->size();
            this->children_index.push_back(index);

            // Set this to terminal if it a masked action (option 1) or if it is the last action (option 2)
            bool terminal = (prior < 0.001) || (action_count >= action_limit - 1);
            if (terminal) {
                this->terminal_children += 1;
            }
            // Add all of the nodes children to the ptr_node_pool
            ptr_node_pool->push_back(CNode(prior, ptr_node_pool, terminal));
        }
    }

    // This method is not currently used because we need to apply noise before taking samples
    // This is done on the python side. Leaving the method here to preserve the core MuZero methods.
    // If you wish to use a pure MuZero implementation, use this noise instead of the python side.
    void CNode::add_exploration_noise(float exploration_fraction, const std::vector<float> &noises) {
        float noise, prior;
        for(int a = 0; a < this->action_num; ++a) {
            noise = noises[a];
            CNode* child = this->get_child(a);

            prior = child->prior;
            child->prior = prior * (1 - exploration_fraction) + noise * exploration_fraction;
        }
    }

    float CNode::compute_mean_q(int isRoot, float parent_q, float discount_factor)
    {
        /*
        Overview:
            Compute the mean q value of the current node.
        Arguments:
            - isRoot: whether the current node is a root node.
            - parent_q: the q value of the parent node.
            - discount_factor: the discount_factor of reward.
        */
        float total_unsigned_q = 0.0;
        int total_visits = 0;
        for (int a = 0; a < this->action_num; ++a)
        {
            CNode *child = this->get_child(a);
            if (child->visit_count > 0)
            {
                float true_reward = child->reward;
                float qsa = true_reward + discount_factor * child->value();
                total_unsigned_q += qsa;
                total_visits += 1;
            }
        }

        float mean_q = 0.0;
        if (isRoot && total_visits > 0)
        {
            mean_q = (total_unsigned_q) / (total_visits);
        }
        else
        {
            mean_q = (parent_q + total_unsigned_q) / (total_visits + 1);
        }
        return mean_q;
    }

    int CNode::expanded() {
        return this->children_index.size() > 0;
    }

    float CNode::value() {
        if(this->visit_count == 0) {
            return 0.0;
        }
        else {
            return this->value_sum / this->visit_count;
        }
    }

    float CNode::qvalue(float discount) {
        return discount * this->value() + this->reward;
    }

    std::vector<int> CNode::get_children_distribution() {
        std::vector<int> distribution;
        distribution.reserve(this->action_num);
        if(this->expanded()) {
            for(int a = 0; a < this->action_num; ++a) {
                CNode* child = this->get_child(a);
                distribution.push_back(child->visit_count);
            }
        }
        return distribution;
    }

    CNode* CNode::get_child(int action) {
        int index = this->children_index[action];
        return &((*(this->ptr_node_pool))[index]);
    }

    //*********************************************************

    CRoots::CRoots() {
        this->root_num = 0;
        this->pool_size = 0;
    }

    // root_num is the number of agents in the batch (NUM_PLAYERS in our base case)
    // pool_size is in place to speed up the vectors and to allocate a given amount of memory at the start
    // Setting this to be the number of samples for now but someone should check if that is correct
    CRoots::CRoots(int root_num, int pool_size) {
        // For whatever reason, print statements do not work inside this function.
        this->root_num = root_num;
        this->pool_size = pool_size;

        this->node_pools.reserve(root_num);
        this->roots.reserve(root_num);

        for(int i = 0; i < root_num; ++i) {
            this->node_pools.push_back(std::vector<CNode>());
            this->node_pools[i].reserve(pool_size);

            this->roots.push_back(CNode(0, &this->node_pools[i], false));
        }
    }

    CRoots::~CRoots() {}

    // This method is not used in our implementation. Leaving it in case we switch back to a pure MuZero implementation
    // This method is only used if you need to apply noise to the input. We apply noise before sampling and before
    // Creating the tree so this method does not get called.
    void CRoots::prepare(float root_exploration_fraction, const std::vector<std::vector<float>> &noises,
                         const std::vector<float> &value_prefixs, const std::vector<std::vector<float>> &policies,
                         const std::vector<int> &action_nums, const std::vector<int> &action_counts,
                         const std::vector<int> &action_limits) {
        for(int i = 0; i < this->root_num; ++i) {
            this->roots[i].expand(0, i, value_prefixs[i], policies[i], action_nums[i], action_counts[i],
                action_limits[i]);
            this->roots[i].add_exploration_noise(root_exploration_fraction, noises[i]);
            this->roots[i].visit_count += 1;
        }
    }

    void CRoots::prepare_no_noise(const std::vector<float> &value_prefixs,
                                  const std::vector<std::vector<float>> &policies,
                                  const std::vector<int> &action_nums,
                                  const std::vector<int> &action_counts,
                                  const std::vector<int> &action_limits) {
        for(int i = 0; i < this->root_num; ++i) {
            this->roots[i].expand(0, i, value_prefixs[i], policies[i], action_nums[i], action_counts[i],
                action_limits[i]);
            this->roots[i].visit_count += 1;
        }
    }

    std::vector<std::vector<int>> CRoots::get_distributions() {
        std::vector<std::vector<int>> distributions;
        distributions.reserve(this->root_num);

        for(int i = 0; i < this->root_num; ++i) {
            distributions.push_back(this->roots[i].get_children_distribution());
        }
        return distributions;
    }

    std::vector<float> CRoots::get_values() {
        std::vector<float> values;
        for(int i = 0; i < this->root_num; ++i) {
            values.push_back(this->roots[i].value());
        }
        return values;
    }

    void cback_propagate(std::vector<CNode*> &search_path, tools::CMinMaxStats &min_max_stats, float value,
                         float discount) {
        // Value from the dynamics network.
        float bootstrap_value = value;
        // How far from root we are.
        int path_len = search_path.size();
        // For each node on our path back to root.
        for(int i = path_len - 1; i >= 0; --i) {
            // Our current node
            CNode* node = search_path[i];
            // Update the value of our node.
            // (bootstrap_value can be negative so this doesn't scale to infinite)
            node->value_sum += bootstrap_value;
            node->visit_count += 1;

            // All of the children are terminal states
            if (node->action_num == node->terminal_children) {
                node->terminal = true;
                if (i > 0) {
                    // update the parent that this is now a terminal state
                    search_path[i-1]->terminal_children += 1;
                }
            }

            // update minimum and maximum
            min_max_stats.update(node->qvalue(discount));

            // update bootstrap for the next value
            bootstrap_value = node->reward + discount * bootstrap_value;
        }
    }

    // create another method here for nodes that we are not updating the value or the reward for since no
    // new node is being created due to the end of the game. This method will go through the search path
    // and update the visit counts but not the reward.
    void cupdate_visit_count(std::vector<CNode*> &search_path) {
        // How far from root we are.
        int path_len = search_path.size();
        // For each node on our path back to root.
        for(int i = path_len - 1; i >= 0; --i) {
            // Our current node
            CNode* node = search_path[i];
            node->visit_count += 1;
        }
    }

    void cbatch_back_propagate(int hidden_state_index_x, float discount, const std::vector<float> &rewards,
                           const std::vector<float> &values, const std::vector<std::vector<float>> &policy,
                           tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results,
                           const std::vector<int> &action_nums, const std::vector<int> &action_counts,
                           const std::vector<int> &action_limits) {
        for (int i = 0; i < results.num; ++i) {
            if (results.nodes[i]->terminal) {
                // Skip backpropagation for terminal roots, as they should not have new nodes
                // This node should only be terminal in the case where the root is terminal
                // All other cases, there is a node in the tree where we don't have a terminal value
                continue;
            }

            // Expand and backpropagate as usual for non-terminal roots
            results.nodes[i]->expand(hidden_state_index_x, i, rewards[i], policy[i], action_nums[i], action_counts[i],
                action_limits[i]);
            cback_propagate(results.search_paths[i], min_max_stats_lst->stats_lst[i], values[i], discount);
        }
    }

    int cselect_child(CNode* root, tools::CMinMaxStats &min_max_stats, int pb_c_base, float pb_c_init, float discount, float mean_q) {
        float max_score = FLOAT_MIN;
        const float epsilon = 0.00001;
        std::vector<int> max_index_lst;

        for(int a = 0; a < root->action_num; ++a) {
            CNode* child = root->get_child(a);

            // find the usb score
            float temp_score = cucb_score(child, min_max_stats, mean_q, root->visit_count - 1, pb_c_base, pb_c_init, discount);
            // compare it to the max score and store index if it is the max
            if(max_score < temp_score) {
                max_score = temp_score;

                max_index_lst.clear();
                max_index_lst.push_back(a);
            }
            else if(temp_score >= max_score - epsilon) {
                max_index_lst.push_back(a);
            }
        }
        int action = 0;
        if(max_index_lst.size() > 0) {
            int rand_index = rand() % max_index_lst.size();
            action = max_index_lst[rand_index];
        }
        return action;
    }

    // values are very high at the start of training compared to the priors so at the start
    // it will go down the tree almost equal to the number of simulations.
    float cucb_score(CNode *child, tools::CMinMaxStats &min_max_stats, float parent_mean_q,
                     float total_children_visit_counts, float pb_c_base, float pb_c_init, float discount) {
        float pb_c = 0.0, prior_score = 0.0, value_score = 0.0;
        // the usb formula
        pb_c = log((total_children_visit_counts + pb_c_base + 1) / pb_c_base) + pb_c_init;
        pb_c *= (sqrt(total_children_visit_counts) / (child->visit_count + 1));

        prior_score = pb_c * child->prior;
        if (child->visit_count == 0) {
            value_score = parent_mean_q;
        }
        else {
            // ensure that the value_score is between 0 and 1, (normally between -300 and 300)
            value_score = child->qvalue(discount);
        }

        value_score = min_max_stats.normalize(value_score);

        // Some testing should occur to see if this is helpful, I think I should delete these lines
        if (value_score < 0) value_score = 0;
        if (value_score > 1) value_score = 1;

//        if (child -> visit_count > 100) {
//            std::cout << "prior score : " << prior_score << " , and value score : " << value_score <<
//                 " and visit_counts : " << child -> visit_count << std::endl;
//                 }
        return prior_score + value_score;
    }

    void cbatch_traverse(CRoots *roots, int pb_c_base, float pb_c_init, float discount,
                     tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results) {
        results.search_lens = std::vector<int>();

        float parent_q = 0.0;

        for (int i = 0; i < results.num; ++i) {
            std::vector<int> last_action;  // Default last action to maintain consistency
            CNode *node = &(roots->roots[i]);
            int is_root = 1;
            int search_len = 0;

            if (node->terminal) {
                // If root is terminal, skip adding to search path but keep consistent outputs
                last_action.push_back(28);
                results.last_actions.push_back(last_action);
                results.search_lens.push_back(search_len);
                results.hidden_state_index_x_lst.push_back(node->hidden_state_index_x);
                results.hidden_state_index_y_lst.push_back(node->hidden_state_index_y);
                results.nodes.push_back(node);
                continue;
            }

            // Traverse as usual for non-terminal roots
            results.search_paths[i].push_back(node);
            while (node->expanded()) {
                float mean_q = node->compute_mean_q(is_root, parent_q, discount);
                is_root = 0;
                parent_q = mean_q;
                int action = cselect_child(node, min_max_stats_lst->stats_lst[i], pb_c_base, pb_c_init, discount, parent_q);
                node = node->get_child(action);
                last_action.push_back(action);
                results.search_paths[i].push_back(node);
                search_len += 1;

                if (node->terminal) {
                    cupdate_visit_count(results.search_paths[i]);
                    results.search_paths[i].clear();
                    last_action.clear();
                    node = &(roots->roots[i]);
                    results.search_paths[i].push_back(node);
                    search_len = 0;
                    continue;
                }
            }

            CNode *parent = results.search_paths[i][results.search_paths[i].size() - 2];
            results.hidden_state_index_x_lst.push_back(parent->hidden_state_index_x);
            results.hidden_state_index_y_lst.push_back(parent->hidden_state_index_y);
            results.last_actions.push_back(last_action);
            results.search_lens.push_back(search_len);
            results.nodes.push_back(node);
        }
    }
}