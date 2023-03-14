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
        this->ptr_node_pool = nullptr;
        this->mappings = std::vector<char*>{};
    }

    CNode::CNode(float prior, std::vector<CNode>* ptr_node_pool) {
        this->prior = prior;
        this->action_num = 0;
        this->visit_count = 0;
        this->value_sum = 0;
        this->reward = 0.0;
        this->ptr_node_pool = ptr_node_pool;
        this->hidden_state_index_x = -1;
        this->hidden_state_index_y = -1;
        this->mappings = std::vector<char*>{};
    }

    CNode::~CNode() {}

    // I am not able to find a definite answer to we use the sampled policy as the base for the prior
    // or if we do a weighted softmax which is what is done below.
    // The alpha-go paper uses .67 as their weight but we appear to use e instead.
    void CNode::expand(int hidden_state_index_x, int hidden_state_index_y, float reward,
                       const std::vector<float> &policy_logits, const std::vector<char*> mappings, int act_num) {
        // Index for finding the hidden state on python side, x is player, y is search path location
        this->hidden_state_index_x = hidden_state_index_x;
        this->hidden_state_index_y = hidden_state_index_y;
        this->reward = reward;
        // Mapping to map 1081 into 3 dimensional action for recurrent inference
        this->mappings = mappings;
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

            // Add all of the nodes children to the ptr_node_pool
            ptr_node_pool->push_back(CNode(prior, ptr_node_pool));
        }
    }

    void CNode::add_exploration_noise(float exploration_fraction, const std::vector<float> &noises){
        float noise, prior;
        for(int a = 0; a < this->action_num; ++a) {
            noise = noises[a];
            CNode* child = this->get_child(a);

            prior = child->prior;
            child->prior = prior * (1 - exploration_fraction) + noise * exploration_fraction;
        }
    }

    int CNode::expanded() {
        int child_num = this->children_index.size();
        if(child_num > 0) {
            return 1;
        }
        else {
            return 0;
        }
    }

    float CNode::value() {
        if(this->visit_count == 0) {
            return 0;
        }
        else{
            return this->value_sum / this->visit_count;
        }
    }

    float CNode::qvalue(float discount) {
        return discount * this->value() + this->reward;
    }

    std::vector<int> CNode::get_children_distribution() {
        std::vector<int> distribution;
        distribution.reserve(this->action_num);
        if(this->expanded()){
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

            this->roots.push_back(CNode(0, &this->node_pools[i]));
        }
    }

    CRoots::~CRoots() {}

    void CRoots::prepare(float root_exploration_fraction, const std::vector<std::vector<float>> &noises,
                         const std::vector<float> &value_prefixs, const std::vector<std::vector<float>> &policies,
                         const std::vector<std::vector<char*>> &mappings, const std::vector<int> &action_nums) {
        for(int i = 0; i < this->root_num; ++i) {
            this->roots[i].expand(0, i, value_prefixs[i], policies[i], mappings[i], action_nums[i]);
            this->roots[i].add_exploration_noise(root_exploration_fraction, noises[i]);
            this->roots[i].visit_count += 1;
        }
    }

    void CRoots::prepare_no_noise(const std::vector<float> &value_prefixs,
                                  const std::vector<std::vector<float>> &policies,
                                  const std::vector<std::vector<char*>> &mappings,
                                  const std::vector<int> &action_nums) {
        for(int i = 0; i < this->root_num; ++i){
            this->roots[i].expand(0, i, value_prefixs[i], policies[i], mappings[i], action_nums[i]);
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
        for(int i = 0; i < this->root_num; ++i){
            values.push_back(this->roots[i].value());
        }
        return values;
    }

    std::vector<int> decode_action(char* &str_action) {
        std::string action(str_action);
        size_t index = action.find("_");
        size_t last_index = 0;
        std::vector<int> element_list;
        while(index != std::string::npos) {
            size_t next_index = action.find("_", index + 1);
            if(next_index > action.size()) {
                next_index = action.size();
            }
            std::string newstr = action.substr(last_index, next_index - index - 1);
            element_list.push_back(std::stoi(newstr));
            last_index = index + 1;
            index = action.find("_", index + 1);
        }
        std::string newstr = action.substr(last_index, action.size() - last_index);
        element_list.push_back(std::stoi(newstr));

        if(index != std::string::npos) {
            std::string newstr = action.substr(index + 1, 1);
            element_list.push_back(std::stoi(newstr));
        }
        while(element_list.size() < 3) {
            element_list.push_back(0);
        }
        return element_list;
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

            // update minimum and maximum
            min_max_stats.update(node->qvalue(discount));

            // update bootstrap for the next value
            bootstrap_value = node->reward + discount * bootstrap_value;
        }
    }

    void cbatch_back_propagate(int hidden_state_index_x, float discount, const std::vector<float> &rewards,
                               const std::vector<float> &values, const std::vector<std::vector<float>> &policy,
                               tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results,
                               std::vector<std::vector<char*>> mappings, const std::vector<int> &action_nums) {
        // For each player
        for(int i = 0; i < results.num; ++i){
            results.nodes[i]->expand(hidden_state_index_x, i, rewards[i], policy[i], mappings[i], action_nums[i]);

            cback_propagate(results.search_paths[i], min_max_stats_lst->stats_lst[i], values[i], discount);
        }
    }

    int cselect_child(CNode* root, tools::CMinMaxStats &min_max_stats, int pb_c_base, float pb_c_init, float discount) {
        float max_score = FLOAT_MIN;
        const float epsilon = 0.0001;
        std::vector<int> max_index_lst;

        for(int a = 0; a < root->action_num; ++a) {
            CNode* child = root->get_child(a);
            // find the usb score
            float temp_score = cucb_score(child, min_max_stats, root->visit_count, pb_c_base, pb_c_init, discount);
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
        if(max_index_lst.size() > 0){
            int rand_index = rand() % max_index_lst.size();
            action = max_index_lst[rand_index];
        }
        return action;
    }

    // values are very high at the start of training compared to the priors so at the start
    // it will go down the tree almost equal to the number of simulations.
    float cucb_score(CNode *child, tools::CMinMaxStats &min_max_stats, float total_children_visit_counts,
                     float pb_c_base, float pb_c_init, float discount) {
        float pb_c = 0.0, prior_score = 0.0, value_score = 0.0;
        // the usb formula
        pb_c = log((total_children_visit_counts + pb_c_base + 1) / pb_c_base) + pb_c_init;
        pb_c *= (sqrt(total_children_visit_counts) / (child->visit_count + 1));

        prior_score = pb_c * child->prior;
        if (child->visit_count == 0) {
            value_score = 0;
        }
        else {
            // ensure that the value_score is between 0 and 1, (normally between -300 and 300)
            value_score = min_max_stats.normalize(child->qvalue(discount));
        }

        // Some testing should occur to see if this is helpful, I think I should delete these lines
        if (value_score < 0) value_score = 0;
        if (value_score > 1) value_score = 1;

        return prior_score + value_score;
    }

    void cbatch_traverse(CRoots *roots, int pb_c_base, float pb_c_init, float discount,
                         tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results) {
        // Last action is a multidimensional action so a vector is required. 3 dimensions in our case
        std::vector<int> last_action{0};

        results.search_lens = std::vector<int>();

        // For each player
        for(int i = 0; i < results.num; ++i) {
            CNode *node = &(roots->roots[i]);
            int search_len = 0;
            // Add current node to search path.
            // This can be a node that has already been explored
            results.search_paths[i].push_back(node);
            while(node->expanded()) {
                // pick the next action to simulate
                int action = cselect_child(node, min_max_stats_lst->stats_lst[i], pb_c_base, pb_c_init, discount);
                // Pick the action from the mappings.
                char* str_action = node->mappings[action];

                // get next node
                node = node->get_child(action);
                // Turn the internal next action into one that the model and environment can understand
                last_action = decode_action(str_action);

                // Add Node to the search path for exploration purposes
                results.search_paths[i].push_back(node);
                search_len += 1;
            }

            // These are all for return values back to the python code. Defined in the cytree.pyx file.
            CNode* parent = results.search_paths[i][results.search_paths[i].size() - 2];
            results.hidden_state_index_x_lst.push_back(parent->hidden_state_index_x);
            results.hidden_state_index_y_lst.push_back(parent->hidden_state_index_y);
            results.last_actions.push_back(last_action);
            results.search_lens.push_back(search_len);
            results.nodes.push_back(node);
        }
    }
}