import copy
import config
from typing import List, Any, Union
import time

import numpy as np
import torch

import core.gumbelctree.gmz_tree as tree_gumbel_muzero

class GumbelMuZeroMCTSCtree(object):
    """
    Overview:
        The C++ implementation of MCTS (batch format) for  Gumbel MuZero.  \
        It completes the ``roots`` and ``search`` methods by calling functions in module ``ctree_gumbel_muzero``, \
        which are implemented in C++.
    Interfaces:
        ``__init__``, ``roots``, ``search``

    ..note::
        The benefit of searching for a batch of nodes at the same time is that \
        it can be parallelized during model inference, thus saving time.
    """
    local_config = dict(
        # (int) The max limitation of simluation times during the simulation.
        num_simulations=50,
        # (float) The alpha value used in the Dirichlet distribution for exploration at the root node of the search tree.
        root_dirichlet_alpha=0.3,
        # (float) The noise weight at the root node of the search tree.
        root_noise_weight=0.25,
        # (float) The maximum change in value allowed during the backup step of the search tree update.
        value_delta_max=0.01,
    )

    def __init__(self, cfg: None) -> None:
        """
        Overview:
            Use the default configuration mechanism. If a user passes in a cfg with a key that matches an existing key \
            in the default configuration, the user-provided value will override the default configuration. Otherwise, \
            the default configuration will be used.
        Arguments:
            - cfg (:obj:`EasyDict`): The configuration passed in by the user.
        """
        # Get the default configuration.
        self.config = cfg
        self.discount_factor = 0.997

    @classmethod
    def roots(cls: int, active_collect_env_num: int, legal_actions: List[Any]) -> "gmz_ctree":
        """
        Overview:
            Initializes a batch of roots to search parallelly later.
        Arguments:
            - root_num (:obj:`int`): the number of the roots in a batch.
            - legal_action_list (:obj:`List[Any]`): the vector of the legal actions for the roots.

        ..note::
            The initialization is achieved by the ``Roots`` class from the ``ctree_gumbel_muzero`` module.
        """
        return tree_gumbel_muzero.Roots(active_collect_env_num, legal_actions)

    def search(self, roots: Any, model: torch.nn.Module, latent_state_roots: List[Any],
               to_play_batch: Union[int, List[Any]]) -> None:
        """
        Overview:
            Do MCTS for a batch of roots. Parallel in model inference. \
            Use C++ to implement the tree search.
        Arguments:
            - roots (:obj:`Any`): a batch of expanded root nodes.
            - latent_state_roots (:obj:`list`): the hidden states of the roots.
            - model (:obj:`torch.nn.Module`): The model used for inference.
            - to_play (:obj:`list`): the to_play list used in in self-play-mode board games.

        .. note::
            The core functions ``batch_traverse`` and ``batch_backpropagate`` are implemented in C++.
        """
        with torch.no_grad():
            model.eval()

            # preparation some constant
            batch_size = roots.num
            device = config.DEVICE
            discount_factor = self.discount_factor
            # the data storage of hidden states: storing the states of all the tree nodes
            latent_state_batch_in_search_path = [latent_state_roots]

            # minimax value storage
            min_max_stats_lst = tree_gumbel_muzero.MinMaxStatsList(batch_size)
            min_max_stats_lst.set_delta(0.01)  # value_delta_max

            for simulation_index in range(50):  # number of simulations
                # In each simulation, we expanded a new node, so in one search,
                # we have ``num_simulations`` num of nodes at most.

                latent_states = []

                # prepare a result wrapper to transport results between python and c++ parts
                results = tree_gumbel_muzero.ResultsWrapper(num=batch_size)

                # traverse to select actions for each root
                # hidden_state_index_x_lst: the first index of leaf node states in hidden_state_pool
                # hidden_state_index_y_lst: the second index of leaf node states in hidden_state_pool
                # the hidden state of the leaf node is hidden_state_pool[x, y]; value prefix states are the same
                """
                MCTS stage 1: Selection
                    Each simulation starts from the internal root state s0, and finishes when the simulation reaches a leaf node s_l.
                    In gumbel muzero, the action at the root node is selected using the Sequential Halving algorithm, while the action 
                    at the interier node is selected based on the completion of the action values.
                    50 - number of simulations, 4 - max_num_considered_actions
                """
                latent_state_index_in_search_path, latent_state_index_in_batch, last_actions, virtual_to_play_batch = \
                    tree_gumbel_muzero.batch_traverse(roots, 50, 4, discount_factor, results, to_play_batch)

                # obtain the states for leaf nodes
                for ix, iy in zip(latent_state_index_in_search_path, latent_state_index_in_batch):
                    latent_states.append(latent_state_batch_in_search_path[ix][iy])

                latent_states = torch.from_numpy(np.asarray(latent_states)).to(device).to(torch.float32)
                # .long() is only for discrete action
                last_actions = np.asarray(last_actions)
                """
                MCTS stage 2: Expansion
                    At the final time-step l of the simulation, the next_latent_state and reward/value_prefix are computed by the dynamics function.
                    Then we calculate the policy_logits and value for the leaf node (next_latent_state) by the prediction function. (aka. evaluation)
                MCTS stage 3: Backup
                    At the end of the simulation, the statistics along the trajectory are updated.
                """
                network_output = model.recurrent_inference(latent_states, last_actions)

                latent_state_batch_in_search_path.append(network_output["hidden_state"].cpu().tolist())
                # tolist() is to be compatible with cpp datatype.
                reward_batch = network_output["reward"].reshape(-1).tolist()
                value_batch = network_output["value"].reshape(-1).tolist()
                policy_logits_batch = network_output["policy_logits"].detach().cpu().tolist()

                # In ``batch_backpropagate()``, we first expand the leaf node using ``the policy_logits`` and
                # ``reward`` predicted by the model, then perform backpropagation along the search path to update the
                # statistics.

                # NOTE: simulation_index + 1 is very important, which is the depth of the current leaf node.
                current_latent_state_index = simulation_index + 1

                # backpropagation along the search path to update the attributes
                tree_gumbel_muzero.batch_back_propagate(
                    current_latent_state_index, discount_factor, reward_batch, value_batch, policy_logits_batch,
                    min_max_stats_lst, results, virtual_to_play_batch
                )
