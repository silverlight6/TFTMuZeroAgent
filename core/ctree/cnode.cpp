#include <iostream>
#include "cnode.h"

namespace tree {
    const std::vector<char*> default_mapping = create_default_mapping();

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
        this->mappings = default_mapping;
    }

    CNode::CNode(float prior, int action_num, std::vector<CNode>* ptr_node_pool) {
        this->prior = prior;
        this->action_num = action_num;

        this->visit_count = 0;
        this->value_sum = 0;
        this->reward = 0.0;
        this->ptr_node_pool = ptr_node_pool;
        this->hidden_state_index_x = -1;
        this->hidden_state_index_y = -1;
        this->mappings = default_mapping;
    }

    CNode::~CNode(){}

    void CNode::expand(int hidden_state_index_x, int hidden_state_index_y, float reward,
                       const std::vector<float> &policy_logits, const std::vector<char*> mappings) {
        // Index for finding the hidden state on python side, x is player, y is search path location
        this->hidden_state_index_x = hidden_state_index_x;
        this->hidden_state_index_y = hidden_state_index_y;
        this->reward = reward;
        // Mapping to map 1081 into 3 dimensional action for recurrent inference
        this->mappings = mappings;

        // Number of actions, normally equal to sample size but can be less if less available actions
        int action_num = this->action_num;
        float temp_policy;
        // sum is a float instead of a tensor since we handle 1 player at a time
        float policy_sum = 0.0;
        std::vector<float> policy(action_num);
        float policy_max = FLOAT_MIN;
        // Find the maximum
        for(int a = 0; a < action_num; ++a){
            if(policy_max < policy_logits[a]){
                policy_max = policy_logits[a];
            }
        }

        // Calculate the sum and create a temp policy with the exp of each
        for(int a = 0; a < action_num; ++a) {
            // exp is e ^ value and since all values are negative, all values in temp_policy are between 0 and 1
            temp_policy = exp(policy_logits[a] - policy_max);
            policy_sum += temp_policy;
            policy[a] = temp_policy;
        }

        float prior;
        std::vector<CNode>* ptr_node_pool = this->ptr_node_pool;
        for(int a = 0; a < action_num; ++a) {
            // Normalizes the array
            prior = policy[a] / policy_sum;
            int index = ptr_node_pool->size();
            this->children_index.push_back(index);

            // Add all of the nodes children to the ptr_node_pool
            ptr_node_pool->push_back(CNode(prior, action_num, ptr_node_pool));
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

    std::vector<int> CNode::get_children_distribution(){
        std::vector<int> distribution;
        if(this->expanded()){
            for(int a = 0; a < this->action_num; ++a){
                CNode* child = this->get_child(a);
                distribution.push_back(child->visit_count);
            }
        }
        return distribution;
    }

    CNode* CNode::get_child(int action){
        int index = this->children_index[action];
        return &((*(this->ptr_node_pool))[index]);
    }



    //*********************************************************

    CRoots::CRoots(){
        this->root_num = 0;
        this->action_num = std::vector<int>{0};
        this->pool_size = 0;
    }

    // root_num is the number of agents in the batch (NUM_PLAYERS in our base case)
    // pool_size is in place to speed up the vectors and to allocate a given amount of memory at the start
    // Setting this to be the number of samples for now but someone should check if that is correct
    CRoots::CRoots(int root_num, std::vector<int> action_num, int pool_size){
        // For whatever reason, print statements do not work inside this function.
        this->root_num = root_num;
        this->action_num = action_num;
        this->pool_size = pool_size;

        this->node_pools.reserve(root_num);
        this->roots.reserve(root_num);

        for(int i = 0; i < root_num; ++i){
            this->node_pools.push_back(std::vector<CNode>());
            this->node_pools[i].reserve(pool_size);

            this->roots.push_back(CNode(0, action_num[i], &this->node_pools[i]));
        }
    }

    CRoots::~CRoots(){}

    void CRoots::prepare(float root_exploration_fraction, const std::vector<std::vector<float>> &noises,
                         const std::vector<float> &value_prefixs, const std::vector<std::vector<float>> &policies,
                         const std::vector<std::vector<char*>> &mappings){
        for(int i = 0; i < this->root_num; ++i) {
            this->roots[i].expand(0, i, value_prefixs[i], policies[i], mappings[i]);
            this->roots[i].add_exploration_noise(root_exploration_fraction, noises[i]);
            this->roots[i].visit_count += 1;
        }
    }

    void CRoots::prepare_no_noise(const std::vector<float> &value_prefixs,
                                  const std::vector<std::vector<float>> &policies,
                                  const std::vector<std::vector<char*>> &mappings){
        for(int i = 0; i < this->root_num; ++i){
            this->roots[i].expand(0, i, value_prefixs[i], policies[i], mappings[i]);
            this->roots[i].visit_count += 1;
        }
    }

    void CRoots::clear(){
        this->node_pools.clear();
        this->roots.clear();
    }

    std::vector<std::vector<int>> CRoots::get_distributions(){
        std::vector<std::vector<int>> distributions;
        distributions.reserve(this->root_num);

        for(int i = 0; i < this->root_num; ++i){
            distributions.push_back(this->roots[i].get_children_distribution());
        }
        return distributions;
    }

    std::vector<float> CRoots::get_values(){
        std::vector<float> values;
        for(int i = 0; i < this->root_num; ++i){
            values.push_back(this->roots[i].value());
        }
        return values;
    }

    std::vector<int> decode_action(char* &str_action) {
        std::string str(str_action);
        char* split_action = strtok(str_action, "_");
        std::vector<int> element_list;
        while(split_action != NULL) {
            element_list.push_back(std::stoi(split_action));
            split_action = strtok(NULL, "_");
        }
        while(element_list.size() < 3) {
            element_list.push_back(0);
        }
        return element_list;
    }

    std::vector<char*> create_default_mapping() {
        std::vector<char*> mapping = std::vector<char*>{};
        // mapping.reserve(1081);
        std::string zero = "0";
        char *copyzero = new char[strlen(&zero[0]) + 1];
        strcpy(copyzero, &zero[0]);
        mapping.push_back(copyzero);

        // Default encodings for the shop.
        for(int i = 0; i < 5; i++) {
            // Create the string that we want to add to the list
            std::string str = "1_" + std::to_string(i);
            // Allocate some memory for the list to live in
            char *copy = new char[strlen(&str[0]) + 1];
            // Copy our string to the allocated memory
            strcpy(copy, &str[0]);
            // Add it to the array.
            // If we don't allocate the memory, it will only push copies of the same string.
            mapping.push_back(copy);
        }

        // Default encodings for the move / sell board / bench
        for(int a = 0; a < 38; a++) {
            for(int b = a; b < 38; b++) {
                if(a == b) {
                    continue;
                }
                if((a > 27) && (b != 38)) {
                    continue;
                }
                std::string str = "2_" + std::to_string(a) + "_" + std::to_string(b);
                char *copy = new char[strlen(&str[0]) + 1];
                strcpy(copy, &str[0]);
                mapping.push_back(copy);
            }
        }

        // Default encodings for the move item
        for(int a = 0; a < 37; a++) {
            for(int b = 0; b < 10; b++) {
                std::string str = "3_" + std::to_string(a) + "_" + std::to_string(b);
                char *copy = new char[strlen(&str[0]) + 1];
                strcpy(copy, &str[0]);
                mapping.push_back(copy);
            }
        }
        std::string four = "4";
        char *copyfour = new char[strlen(&four[0]) + 1];
        strcpy(copyfour, &four[0]);
        mapping.push_back(copyfour);
        std::string five = "5";
        char *copyfive = new char[strlen(&five[0]) + 1];
        strcpy(copyfive, &five[0]);
        mapping.push_back(copyfive);
        
        return mapping;
    }

    void cback_propagate(std::vector<CNode*> &search_path, tools::CMinMaxStats &min_max_stats, float value,
                         float discount) {
        // Value from the dynamics network.
        float bootstrap_value = value;
        // How far from root we are.
        int path_len = search_path.size();
        // For each node on our path back to root.
        for(int i = path_len - 1; i >= 0; --i){
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
        // Not sure if this line is needed or not. It's not on the python side
        // min_max_stats.clear();
    }

    void cbatch_back_propagate(int hidden_state_index_x, float discount, const std::vector<float> &rewards,
                               const std::vector<float> &values, const std::vector<std::vector<float>> &policy,
                               tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results,
                               std::vector<std::vector<char*>> mappings){
        // For each player
        for(int i = 0; i < results.num; ++i){
            results.nodes[i]->expand(hidden_state_index_x, i, rewards[i], policy[i], mappings[i]);

            cback_propagate(results.search_paths[i], min_max_stats_lst->stats_lst[i], values[i], discount);
        }
    }

    int cselect_child(CNode* root, tools::CMinMaxStats &min_max_stats, int pb_c_base, float pb_c_init, float discount) {
        float max_score = FLOAT_MIN;
        int action_idx = -1;
        for(int a = 0; a < root->action_num; ++a){
            CNode* child = root->get_child(a);
            // find the usb score
            float temp_score = cucb_score(child, min_max_stats, root->visit_count - 1,
                                          pb_c_base, pb_c_init, discount);
            // compare it to the max score and store index if it is the max
            if(max_score < temp_score){
                max_score = temp_score;
                action_idx = a;
            }
        }
        return action_idx;
    }

    float cucb_score(CNode *child, tools::CMinMaxStats &min_max_stats, float total_children_visit_counts,
                     float pb_c_base, float pb_c_init, float discount){
        float pb_c = 0.0, prior_score = 0.0, value_score = 0.0;
        // the usb formula
        pb_c = log((total_children_visit_counts + pb_c_base + 1) / pb_c_base) + pb_c_init;
        pb_c *= (sqrt(total_children_visit_counts) / (child->visit_count + 1));

        prior_score = pb_c * child->prior;
        if (child->visit_count == 0){
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
                         tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results){

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