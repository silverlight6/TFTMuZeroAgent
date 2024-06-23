import gymnasium as gym
import time

from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core.models.base import ACTOR, CRITIC, ENCODER_OUT, STATE_OUT
from ray.rllib.core.rl_module.rl_module import RLModule, RLModuleConfig
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.torch_utils import FLOAT_MIN
from typing import Mapping, Any

torch, nn = try_import_torch()

ModuleID = str


# Not 100% sure this class is even needed but keeping just in case.
class ActionMaskRLMBase(RLModule):
    def __init__(self, config: RLModuleConfig):
        if not isinstance(config.observation_space, gym.spaces.Dict):
            raise ValueError(
                "This model requires the environment to provide a "
                "gym.spaces.Dict observation space."
            )

        super().__init__(config)


class TorchActionMaskRLM(ActionMaskRLMBase, PPOTorchRLModule):
    """
    Description - Overloading the ray base torch reinforcement learning module object to support action masks and
                    Ray Modelv2 Models.
    """
    def _forward_inference(self, batch, **kwargs):
        return self.mask_forward_fn_torch(self._local_forward_inference, batch, **kwargs)

    def _forward_train(self, batch, *args, **kwargs):
        return self.mask_forward_fn_torch(self._local_forward_train, batch, **kwargs)

    def _forward_exploration(self, batch, *args, **kwargs):
        return self.mask_forward_fn_torch(self._local_forward_exploration, batch, **kwargs)

    def mask_forward_fn_torch(self, forward_fn, batch, **kwargs):
        self._check_batch(batch)
        # Remote the action mask so we can reshape it as needed
        action_mask = batch[SampleBatch.OBS]["action_mask"]

        # Call the model
        outputs = forward_fn(batch, **kwargs)
        logits = outputs[SampleBatch.ACTION_DIST_INPUTS]
        action_mask_shape = action_mask.shape
        # Reshape so that it is batch by 28 * 12 to complement the shape that ray requires
        # for multi-categorical action spaces
        action_mask = torch.reshape(action_mask, (action_mask_shape[0], action_mask_shape[1] * action_mask_shape[2]))

        # Mask
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        outputs[SampleBatch.ACTION_DIST_INPUTS] = masked_logits

        return outputs

    # Overloading these methods so I can call them as they want to be called with Ray Modelv2
    def _local_forward_inference(self, batch: NestedDict) -> Mapping[str, Any]:
        output = {}

        encoder_outs, _ = self.encoder(batch)

        # Actions
        action_logits, _ = self.pi({"obs": encoder_outs[ENCODER_OUT][ACTOR]})
        output[SampleBatch.ACTION_DIST_INPUTS] = action_logits

        return output

    def _local_forward_exploration(self, batch: NestedDict) -> Mapping[str, Any]:
        """PPO forward pass during exploration.
        Besides the action distribution, this method also returns the parameters of the
        policy distribution to be used for computing KL divergence between the old
        policy and the new policy during training.
        """
        output = {}

        # Shared encoder
        encoder_outs, _ = self.encoder(batch)

        # Value head
        vf_out, _ = self.vf({"obs": encoder_outs[ENCODER_OUT][CRITIC]})
        output[SampleBatch.VF_PREDS] = vf_out.squeeze(-1)

        # Policy head
        action_logits, _ = self.pi({"obs": encoder_outs[ENCODER_OUT][ACTOR]})
        output[SampleBatch.ACTION_DIST_INPUTS] = action_logits

        return output

    def _local_forward_train(self, batch: NestedDict) -> Mapping[str, Any]:
        output = {}

        # Shared encoder
        encoder_outs, _ = self.encoder(batch)

        # Value head
        vf_out, _ = self.vf({"obs": encoder_outs[ENCODER_OUT][CRITIC]})
        output[SampleBatch.VF_PREDS] = vf_out.squeeze(-1)

        # Policy head
        action_logits, _ = self.pi({"obs": encoder_outs[ENCODER_OUT][ACTOR]})
        output[SampleBatch.ACTION_DIST_INPUTS] = action_logits

        return output

    # overriding another method to avoid errors
    def _check_batch(self, batch):
        """Check whether the batch contains the required keys."""
        if "action_mask" not in batch[SampleBatch.OBS]:
            raise ValueError(
                "Action mask not found in observation. This model requires "
                "the environment to provide observations that include an "
                "action mask (i.e. an observation space of the Dict space "
                "type that looks as follows: \n"
                "{'action_mask': Box(0.0, 1.0, shape=(self.action_space.n,)),"
                "'observations': <observation_space>}"
            )
        if "observations" not in batch[SampleBatch.OBS]:
            raise ValueError(
                "Observations not found in observation.This model requires "
                "the environment to provide observations that include a "
                " (i.e. an observation space of the Dict space "
                "type that looks as follows: \n"
                "{'action_mask': Box(0.0, 1.0, shape=(self.action_space.n,)),"
                "'observations': <observation_space>}"
            )
