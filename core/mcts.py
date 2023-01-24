import torch
import config

import numpy as np
import core.ctree.cytree as tree

from torch.cuda.amp import autocast as autocast


class MCTS(object):
    def __init__(self, config):
        self.config = config

    def search(self, roots, model, hidden_state_roots, reward_hidden_roots):
        """Do MCTS for the roots (a batch of root nodes in parallel). Parallel in model inference
        Parameters
        ----------
        we have roots 
        roots: Any 
            a batch of expanded root nodes
        network_output["hidden_state"][to_play]
        hidden_state_roots: list
            the hidden states of the roots
         network_output["hidden_state"][to_play]
        reward_hidden_roots: list
            the value prefix hidden states in LSTM of the roots
        """
        with torch.no_grad():
            model.eval()

            # preparation
            num = len(roots)
            discount = config.DISCOUNT  # we have these 
            # the data storage of hidden states: storing the states of all the tree nodes
            # 1 x batch x 64
            # the data storage of value prefix hidden states in LSTM
            # the index of each layer in the tree
            hidden_state_index_x = 0
            # minimax value storage
            min_max_stats = [MinMaxStats(config.MINIMUM_REWARD, config.MAXIMUM_REWARD) for _ in range(config.NUM_PLAYERS)]
            min_max_stats_lst = tree.MinMaxStatsList(num)
            min_max_stats_lst.set_delta(config.MAXIUMUM_REWARD*2)  # config.MINIMUM_REWARD *2 (double check)
            horizons = 1  # self.config.lstm_horizon_len

            for index_simulation in range(self.config.num_simulations):
                # prepare a result wrapper to transport results between python and c++ parts
                results = tree.ResultsWrapper(num)
                # traverse to select actions for each root
                # hidden_state_index_x_lst: the first index of leaf node states in hidden_state_pool
                # hidden_state_index_y_lst: the second index of leaf node states in hidden_state_pool
                # the hidden state of the leaf node is hidden_state_pool[x, y]; value prefix states are the same
                # obtain the search horizon for leaf nodes
                search_lens = results.get_search_len()

                # hidden_states = torch.from_numpy(np.asarray(hidden_states)).to(device).float()
                # last_actions = torch.from_numpy(np.asarray(last_actions)).to(device).unsqueeze(1).long() 

                # evaluation for leaf nodes
                history = [ActionHistory(action[i]) for i in range(config.NUM_PLAYERS)]
                node = roots
                search_path = [[node[i]] for i in range(config.NUM_PLAYERS)]
                
                # There is a chance I am supposed to check if the tree for the non-main-branch
                # Decision paths (axis 1-4) should be expanded. I am currently only expanding on the
                # main decision axis.
                for i in range(config.NUM_PLAYERS):
                    while node[i].expanded():
                        action[i], node[i] = self.select_child(node[i], min_max_stats[i])
                        history[i].add_action(action[i])
                        search_path[i].append(node[i])
                
                # Inside the search tree we use the dynamics function to obtain the next
                # hidden state given an action and the previous hidden state.
                parent = [search_path[i][-2] for i in range(config.NUM_PLAYERS)]
                hidden_state = np.asarray([parent[i].hidden_state for i in range(config.NUM_PLAYERS)])
                last_action = np.asarray([history[i].last_action() for i in range(config.NUM_PLAYERS)])               
                
                network_output = self.network.recurrent_inference(hidden_state, last_action) #11.05s 
                #network_output = "temp"
                #dont need (have our own implementation )
                # if self.config.amp_type == 'torch_amp':
                #     with autocast():
                #         network_output = model.recurrent_inference(hidden_states, (hidden_states_c_reward, hidden_states_h_reward), last_actions)
                # else:
                #     network_output = model.recurrent_inference(hidden_states, (hidden_states_c_reward, hidden_states_h_reward), last_actions)
                value_prefix_pool = network_output["value_logits"].reshape(-1).tolist()
                value_pool = network_output["value"].reshape(-1).tolist()
                policy_logits_pool = network_output["policy_logits"].tolist()


                # reset 0
                # reset the hidden states in LSTM every horizon steps in search
                # only need to predict the value prefix in a range (eg: s0 -> s5)
                assert horizons > 0
                reset_idx = (np.array(search_lens) % horizons == 0)
                assert len(reset_idx) == num
                is_reset_lst = reset_idx.astype(np.int32).tolist()
                hidden_state_index_x += 1

                # backpropagation along the search path to update the attributes
                tree.batch_back_propagate(hidden_state_index_x, discount,
                                          value_prefix_pool, value_pool, policy_logits_pool,
                                          min_max_stats_lst, results, is_reset_lst)