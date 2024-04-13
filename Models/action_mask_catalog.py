import gymnasium as gym
from dataclasses import dataclass
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.core.models.base import Model
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.utils.annotations import ExperimentalAPI
from typing import Optional


class ActionMaskCatalog(PPOCatalog):

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        model_config_dict: dict,
    ):
        """Initializes the Action Mask Catalog which is overwriting the PPO Catalog.

        Args:
            observation_space: The observation space of the Encoder.
            action_space: The action space for the Pi Head.
            model_config_dict: The model config to use.
        """
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            model_config_dict=model_config_dict,
        )

        # Most of these are so the encoder, policy function, and value function are all seeing the same configs.
        self.hidden_state = model_config_dict["custom_model_config"]["hidden_state_size"]
        self.num_layers = model_config_dict["custom_model_config"]["num_hidden_layers"]
        self.model_config_dict = model_config_dict
        # Have yet to find a better way to pass this information.
        self.model_config_dict["device"] = "cuda"

        # Same goes for this if someone wants to expand this for JAX eventually.
        self.framework = "torch"

        self.vf_head_config = ActionMaskModelVFConfig(
            act_space=action_space,
            obs_space=observation_space,
            output_size=1,
            model_dict=model_config_dict
        )

    @classmethod
    def _get_encoder_config(
        cls,
        observation_space: gym.Space,
        model_config_dict: dict,
        action_space: gym.Space = None,
        view_requirements=None,
    ) -> ModelConfig:

        model_config_dict = {**MODEL_DEFAULTS, **model_config_dict}
        model_config_dict["device"] = "cuda"

        return ActionMaskModelEncoderConfig(
            act_space=action_space,
            obs_space=observation_space,
            output_size=model_config_dict["custom_model_config"]["hidden_state_size"],
            model_dict=model_config_dict
        )

    def build_actor_critic_encoder(self, framework: str):
        return self._get_encoder_config(self.observation_space,
                                        self.model_config_dict,
                                        self.action_space).build(framework=framework)

    def build_vf_head(self, framework: str) -> Model:
        """Builds the value function head.

        The default behavior is to build the head from the vf_head_config.
        This can be overridden to build a custom value function head as a means of
        configuring the behavior of a PPORLModule implementation.

        Args:
            framework: The framework to use. Either "torch" or "tf2".

        Returns:
            The value function head.
        """
        return self.vf_head_config.build(framework=framework)

    def build_pi_head(self, framework: str) -> Model:
        self.pi_head_config = ActionMaskModelPConfig(
            act_space=self.action_space,
            obs_space=self.observation_space,
            model_dict=self.model_config_dict
        )
        return self.pi_head_config.build(framework=framework)


@ExperimentalAPI
@dataclass
class ActionMaskModelEncoderConfig(ModelConfig):
    hidden_layer_use_bias: bool = True
    hidden_layer_activation: str = "relu"
    hidden_layer_use_layernorm: bool = False
    output_size: int = 256

    # Probably could delete most of these extra variables and it would be fine.
    # Optional last output layer with - possibly - different activation and use_bias
    # settings.
    output_layer_dim: Optional[int] = None
    output_layer_use_bias: bool = True
    output_layer_activation: str = "linear"
    shared: bool = True

    act_space: gym.Space = None
    obs_space: gym.Space = None
    model_dict: dict = None
    device: str = 'cuda'

    @property
    def output_dims(self):
        if self.output_layer_dim is None and not self.output_size:
            raise ValueError(
                "If `output_layer_dim` is None, you must specify at least one hidden "
                "layer dim, e.g. `hidden_layer_dims=[32]`!"
            )

        # Infer `output_dims` automatically.
        return self.output_layer_dim or self.output_size,

    def _validate(self, framework: str = "torch"):
        """Makes sure that settings are valid."""
        if self.input_dims is not None and len(self.input_dims) != 1:
            raise ValueError(
                f"`input_dims` ({self.input_dims}) of MLPConfig must be 1D, "
                "e.g. `[32]`!"
            )
        if len(self.output_dims) != 1:
            raise ValueError(
                f"`output_dims` ({self.output_dims}) of _MLPConfig must be "
                "1D, e.g. `[32]`! This is an inferred value, hence other settings might"
                " be wrong."
            )

        # Call these already here to catch errors early on.
        get_activation_fn(self.hidden_layer_activation, framework=framework)
        get_activation_fn(self.output_layer_activation, framework=framework)

    def build(self, framework: str = "torch"):
        self._validate(framework=framework)
        from Models.action_mask_model import TorchActionMaskEncoderModel
        return TorchActionMaskEncoderModel(action_space=self.act_space,
                                           obs_space=self.obs_space,
                                           num_outputs=self.output_size,
                                           model_config=self.model_dict,
                                           name="ActionMaskEncoderModel")


@ExperimentalAPI
@dataclass
class ActionMaskModelVFConfig(ModelConfig):
    output_size: int = 1
    act_space: gym.Space = None
    obs_space: gym.Space = None
    model_dict: dict = None
    device: str = 'cuda'

    @property
    def output_dims(self):
        # Value function so it is always going to be 1
        return 1

    def _validate(self, framework: str = "torch"):
        """Required as part of the parent class but I have nothing to validate."""
        ...

    def build(self, framework: str = "torch"):
        self._validate(framework=framework)
        from Models.action_mask_model import TorchActionMaskValueModel
        return TorchActionMaskValueModel(action_space=self.act_space,
                                         obs_space=self.obs_space,
                                         num_outputs=self.output_size,
                                         model_config=self.model_dict,
                                         name="ActionMaskVFModel")


# P stands for policy here.
@dataclass
class ActionMaskModelPConfig(ModelConfig):
    act_space: gym.Space = None
    obs_space: gym.Space = None
    model_dict: dict = None
    device: str = 'cuda'
    output_size: int = 336  # 12 * 28

    # Honestly nothing to validate.
    def _validate(self, framework: str = "torch"):
        ...

    def build(self, framework: str = "torch") -> "Model":
        self._validate(framework=framework)

        from Models.action_mask_model import TorchActionMaskPolicyModel
        return TorchActionMaskPolicyModel(action_space=self.act_space,
                                          obs_space=self.obs_space,
                                          num_outputs=self.output_size,
                                          model_config=self.model_dict,
                                          name="ActionMaskPolicyModel")
