import torch
from Core.MCTS_Trees.MCTS_Util import ValueEncoder, inverse_contractive_mapping
from Core.TorchComponents.dynamics_models import AtariDynamicsNetwork as DynamicsNetwork
from Core.TorchComponents.prediction_models import AtariValueNetwork as ValueNetwork, \
    AtariPolicyNetwork as PolicyNetwork
from Core.TorchComponents.representation_models import AtariRepresentationNetwork as RepresentationNetwork
from Core.TorchModels.abstract_model import AbstractNetwork

class AtariMuZero(AbstractNetwork):
    def __init__(
            self,
            num_frames: int = 4,
            hidden_channels: int = 256,
            kernel_size: int = 3,
            rep_blocks: int = 4,
            dyn_blocks: int = 16,
            device: str = "cpu",
    ):
        super().__init__()

        self.representation_network = RepresentationNetwork(
            in_channels=num_frames * 3,
            hidden_channels=hidden_channels,
            num_blocks=rep_blocks,
            kernel_size=kernel_size,
            device=device,
        )
        self.dynamics_network = DynamicsNetwork(
            hidden_channels=hidden_channels,
            action_space=17,
            num_blocks=dyn_blocks,
            kernel_size=kernel_size,
            device=device,
        )
        self.policy_network = PolicyNetwork(
            hidden_channels=hidden_channels, device=device
        )
        self.value_network = ValueNetwork(
            hidden_channels=hidden_channels, support_size=601, device=device
        )
        self.reward_network = ValueNetwork(
            hidden_channels=hidden_channels, support_size=601, device=device
        )

        map_min = torch.tensor(-300.0, dtype=torch.float32)
        map_max = torch.tensor(300.0, dtype=torch.float32)
        self.value_encoder = ValueEncoder(
            *tuple(map(inverse_contractive_mapping, (map_min, map_max))),
            0,
        )
        self.reward_encoder = ValueEncoder(
            *tuple(map(inverse_contractive_mapping, (map_min, map_max))),
            0,
        )

    def initial_inference(self, x, training=False):
        batch_size = x.shape[0]

        hidden_state = self.representation_network(x)
        value_logits = self.value_network(hidden_state)
        policy_logits = self.policy_network(hidden_state)
        reward = torch.zeros((batch_size,))
        value = self.value_encoder.decode_softmax(value_logits)

        outputs = {
            "value": value,
            "value_logits": value_logits,
            "reward": reward,
            "policy_logits": policy_logits,
            "hidden_state": hidden_state,
        }

        return outputs

    def recurrent_inference(self, encoded_state, action, training=False):
        next_hidden_state = self.dynamics_network(encoded_state, action)
        reward_logits = self.reward_network(next_hidden_state)

        value_logits = self.value_network(next_hidden_state)
        policy_logits = self.policy_network(next_hidden_state)

        reward = self.reward_encoder.decode_softmax(reward_logits)
        value = self.value_encoder.decode_softmax(value_logits)

        outputs = {
            "value": value,
            "value_logits": value_logits,
            "reward": reward,
            "reward_logits": reward_logits,
            "policy_logits": policy_logits,
            "hidden_state": next_hidden_state,
        }

        return outputs