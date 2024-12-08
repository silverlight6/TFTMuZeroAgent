import config
import time
from typing import List, Dict, Tuple
import numpy as np
import torch
from Models.GumbelModels.MCTSCtree import GumbelMuZeroMCTSCtree as MCTSCtree
from scipy.stats import entropy

class GumbelMuZero:
    def __init__(self, network, model_config):
        self.network = network
        self.model_config = model_config
        self.discount_factor = config.DISCOUNT
        self.action_space_size = config.POLICY_HEAD_SIZE
        self._collect_model = self.network
        self._mcts_collect = MCTSCtree(self.model_config)
        self._collect_mcts_temperature = 1

    def policy(self, data: Dict, temperature: float = 1, to_play: List = [-1], ):
        """
        Overview:
            The forward function for collecting data in collect mode. Use model to execute MCTS search.
            Choosing the action through sampling during the collect mode.
        Arguments:
            - data (:obj:`dict`): The input data, i.e. the observation and the action mask.
            - temperature (:obj:`float`): The temperature of the policy.
            - to_play (:obj:`int`): The player to play.
            - ready_env_id (:obj:`list`): The id of the env that is ready to collect.
        Shape:
            - data (:obj:`dict`):
                - observation - dictionary of dictionaries
                - action mask - numpy array (55 x 38)
            - temperature: :math:`(1, )`.
            - to_play: :math:`(N, 1)`, where N is the number of collect_env.
            - ready_env_id: None
        Returns:
            - output (:obj:`Dict[int, Any]`): Dict type data, the keys including ``action``, ``distributions``, \
                ``visit_count_distribution_entropy``, ``value``, ``roots_completed_value``, ``improved_policy_probs``, \
                ``pred_value``, ``policy_logits``.
        """
        self._collect_model.eval()
        self._collect_mcts_temperature = temperature
        # batch_size = data["observations"]["shop"].shape[0]
        batch_size = data["action_mask"].shape[0]
        ready_env_id = np.arange(batch_size)

        with torch.no_grad():
            # data shape [B, S x C, W, H], e.g. {Tensor:(B, 12, 96, 96)}
            network_output = self._collect_model.initial_inference(data["observations"])

            pred_values = network_output["value"].detach().cpu().numpy()
            reward = network_output["reward"].detach().cpu().numpy()
            latent_state_roots = network_output["hidden_state"].detach().cpu().numpy()
            policy_logits = network_output["policy_logits"].detach().cpu().numpy().tolist()
            action_mask = np.reshape(data["action_mask"], (batch_size, -1))

            legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1.0] for j in range(batch_size)]

            # the only difference between collect and eval is the dirichlet noise
            noises = [
                np.random.dirichlet([self.model_config.ROOT_DIRICHLET_ALPHA] * int(sum(action_mask[j]))
                                    ).astype(np.float32).tolist() for j in range(batch_size)
            ]
            roots = MCTSCtree.roots(batch_size, legal_actions)
            print(f"noises {noises}, reward {list(reward)}, values {list(pred_values)}, policy_logits {policy_logits}, "
                  f"to_play {to_play}")
            time.sleep(2)
            roots.prepare(self.model_config.ROOT_EXPLORATION_FRACTION, noises, list(reward), list(pred_values),
                          policy_logits, to_play)
            self._mcts_collect.search(roots, self._collect_model, latent_state_roots, to_play)

            # list of list, shape: ``{list: batch_size} -> {list: action_space_size}``
            roots_visit_count_distributions = roots.get_distributions()
            roots_values = roots.get_values()  # shape: {list: batch_size}

            # ==============================================================
            # The core difference between GumbelMuZero and MuZero
            # ==============================================================
            # Gumbel MuZero selects the action according to the improved policy
            # new policy constructed with completed Q in gumbel muzero
            roots_improved_policy_probs = roots.get_policies(self.discount_factor, self.action_space_size)
            roots_improved_policy_probs = np.array(roots_improved_policy_probs)
            actions = []
            target_policies = []

            for i, env_id in enumerate(ready_env_id):
                distributions, value, improved_policy_probs = roots_visit_count_distributions[i], roots_values[i], \
                    roots_improved_policy_probs[i]

                # NOTE: Only legal actions possess visit counts, so the ``action_index_in_legal_action_set`` represents
                # the index within the legal action set, rather than the index in the entire action set.
                # action_index_in_legal_action_set, visit_count_distribution_entropy = self.select_action(
                #     distributions, temperature=self._collect_mcts_temperature, deterministic=False
                # )
                # NOTE: Convert the ``action_index_in_legal_action_set`` to the corresponding ``action`` in the
                # entire action set.
                # action = np.where(action_mask[i] == 1.0)[0][action_index_in_legal_action_set]
                valid_value = np.where(action_mask[i] == 1.0, improved_policy_probs, 0.0)
                action = np.argmax([v for v in valid_value])
                actions.append(action)
                target_policies.append(improved_policy_probs)

        return actions, target_policies, roots_values

    def _init_eval(self) -> None:
        """
        Overview:
            Evaluate mode init method. Called by ``self.__init__``. Initialize the eval model and MCTS utils.
        """
        self._eval_model = self.network
        self._mcts_eval = MCTSCtree(self.model_config)

    def _forward_eval(self, data: torch.Tensor, action_mask: list, to_play: int = -1, ready_env_id: np.array = None, ):
        """
        Overview:
            The forward function for evaluating the current policy in eval mode. Use model to execute MCTS search.
            Choosing the action with the highest value (argmax) rather than sampling during the eval mode.
        Arguments:
            - data (:obj:`torch.Tensor`): The input data, i.e. the observation.
            - action_mask (:obj:`list`): The action mask, i.e. the action that cannot be selected.
            - to_play (:obj:`int`): The player to play.
            - ready_env_id (:obj:`list`): The id of the env that is ready to collect.
        Shape:
            - data (:obj:`torch.Tensor`):
                - For Atari, :math:`(N, C*S, H, W)`, where N is the number of collect_env, C is the number of channels, \
                    S is the number of stacked frames, H is the height of the image, W is the width of the image.
                - For lunarlander, :math:`(N, O)`, where N is the number of collect_env, O is the observation space size.
            - action_mask: :math:`(N, action_space_size)`, where N is the number of collect_env.
            - to_play: :math:`(N, 1)`, where N is the number of collect_env.
            - ready_env_id: None
        Returns:
            - output (:obj:`Dict[int, Any]`): Dict type data, the keys including ``action``, ``distributions``, \
                ``visit_count_distribution_entropy``, ``value``, ``pred_value``, ``policy_logits``.
        """
        self._eval_model.eval()
        active_eval_env_num = data.shape[0]
        ready_env_id = np.arange(active_eval_env_num)
        with torch.no_grad():
            # data shape [B, S x C, W, H], e.g. {Tensor:(B, 12, 96, 96)}
            network_output = self._collect_model.initial_inference(data)

            if not self._eval_model.training:
                # if not in training, obtain the scalars of the value/reward
                pred_values = network_output["value"].detach().cpu().numpy()  # shape（B, 1）
                latent_state_roots = network_output["hidden_state"].detach().cpu().numpy()
                policy_logits = network_output["policy_logits"].detach().cpu().numpy().tolist()  # list shape（B, A）

            legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(active_eval_env_num)]
            roots = MCTSCtree.roots(active_eval_env_num, legal_actions)
            roots.prepare_no_noise(network_output["reward"], list(pred_values), policy_logits, to_play)
            self._mcts_eval.search(roots, self._eval_model, latent_state_roots, to_play)

            # list of list, shape: ``{list: batch_size} -> {list: action_space_size}``
            roots_visit_count_distributions = roots.get_distributions()
            roots_values = roots.get_values()  # shape: {list: batch_size}

            # ==============================================================
            # The core difference between GumbelMuZero and MuZero
            # ==============================================================
            # Gumbel MuZero selects the action according to the improved policy
            roots_improved_policy_probs = roots.get_policies(self.discount_factor, self.action_space_size)
            roots_improved_policy_probs = np.array(roots_improved_policy_probs)
            actions = []
            target_policies = []

            for i, env_id in enumerate(ready_env_id):
                distributions, value, improved_policy_probs = roots_visit_count_distributions[i], roots_values[i], \
                    roots_improved_policy_probs[i]
                # NOTE: Only legal actions possess visit counts, so the ``action_index_in_legal_action_set`` represents
                # the index within the legal action set, rather than the index in the entire action set.
                # Setting deterministic=True implies choosing the action with the highest value (argmax) rather than
                # sampling during the evaluation phase.
                valid_value = np.where(action_mask[i] == 1.0, improved_policy_probs, 0.0)
                action = np.argmax([v for v in valid_value])
                actions.append(action)
                target_policies.append(distributions)
                print(f"action {action}")

        return actions, target_policies, [], roots_values

    def _monitor_vars_learn(self) -> List[str]:
        """
        Overview:
            Register the variables to be monitored in learn mode. The registered variables will be logged in
            tensorboard according to the return value ``_forward_learn``.
        """
        return [
            'collect_mcts_temperature',
            'cur_lr',
            'weighted_total_loss',
            'total_loss',
            'policy_loss',
            'reward_loss',
            'value_loss',
            'consistency_loss',
            'policy_entropy',
            'value_priority',
            'target_reward',
            'target_value',
            'predicted_rewards',
            'predicted_values',
            'transformed_target_reward',
            'transformed_target_value',
            'total_grad_norm_before_clip',
        ]

    def select_action(self, visit_counts: np.ndarray,
                      temperature: float = 1,
                      deterministic: bool = True) -> Tuple[np.int64, np.ndarray]:
        """
        Overview:
            Select action from visit counts of the root node.
        Arguments:
            - visit_counts (:obj:`np.ndarray`): The visit counts of the root node.
            - temperature (:obj:`float`): The temperature used to adjust the sampling distribution.
            - deterministic (:obj:`bool`):  Whether to enable deterministic mode in action selection. True means to \
                select the argmax result, False indicates to sample action from the distribution.
        Returns:
            - action_pos (:obj:`np.int64`): The selected action position (index).
            - visit_count_distribution_entropy (:obj:`np.ndarray`): The entropy of the visit count distribution.
        """
        action_probs = [visit_count_i ** (1 / temperature) for visit_count_i in visit_counts]
        action_probs = [x / sum(action_probs) for x in action_probs]

        if deterministic:
            action_pos = np.argmax([v for v in visit_counts])
        else:
            action_pos = np.random.choice(len(visit_counts), p=action_probs)

        visit_count_distribution_entropy = entropy(action_probs, base=2)
        return action_pos, visit_count_distribution_entropy
