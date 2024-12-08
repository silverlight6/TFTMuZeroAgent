import ale_py
import config
import gymnasium as gym
import numpy as np
import torch

from Core.MCTS_Trees.gumbel_MCTS import GumbelMuZero
from Core.TorchModels.Atari_MuZero_model import AtariMuZero
from Core.Trainers.gumbel_trainer import Trainer
from datetime import datetime
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, TimeLimit
from gymnasium.vector import AsyncVectorEnv
from TestInterface.test_global_buffer import GlobalBuffer
from TestInterface.test_replay_muzero_buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

class Atari_Tests:
    def __init__(self):
        ...

    # TODO: Tomorrow. Change out the replay buffer to be a list of buffers and reset the buffer after game ends
    # TODO: Exit when the global buffer is basically full.
    @staticmethod
    def rollout(env, preprocess_obs, global_buffer, agent, buffers):
        for buffer in buffers:
            buffer.reset()
        obs, info = env.reset()

        while global_buffer.buffer_len() < 10000:
            # Get action
            current_obs = preprocess_obs(obs, config.DEVICE)

            actions, target_policies, roots_values = agent.policy(data=current_obs)

            # Step in Environment
            next_obs, rewards, terminated, truncated, info = env.step(actions)

            # Add to buffer
            for i in range(len(actions)):
                buffers[i].store_replay_buffer(current_obs["observations"][i].detach().cpu(), actions[i], rewards[i],
                                               target_policies[i], roots_values[i])
                if terminated[i] or truncated[i]:
                    buffers[i].print_reward()
                    buffers[i].store_global_buffer()
                    buffers[i].reset()

            obs = next_obs

    def breakout(self):

        # -- Load Config
        model_config = config.ModelConfig()
        model_config.DEFAULT_MASK = False
        model_config.POLICY_HEAD_SIZE = 17

        # -- Setup Environment
        gym.register_envs(ale_py)

        def breakout_env():
            env = gym.make("ALE/Breakout-v5", frameskip=1)
            env = AtariPreprocessing(
                env,
                noop_max=10,
                frame_skip=4,
                terminal_on_life_loss=True,
                screen_size=96,
                grayscale_obs=False,
                grayscale_newaxis=False,
                scale_obs=True,
            )
            env = FrameStackObservation(env, stack_size=4)
            env = TimeLimit(env, max_episode_steps=5000)
            return env

        # -- Observation Utilities
        def create_breakout_action_mask() -> np.array:
            action_mask = np.zeros((config.NUM_ENVS, 17))
            # only the first 4 actions are legal
            action_mask[:, :4] = 1
            return action_mask

        def preprocess_obs(obs: np.ndarray, device: str = "cpu"):
            B, N, H, W, C = obs.shape
            torch_obs = (
                torch.from_numpy(obs)
                .reshape(B, H, W, N * C)
                .permute(0, 3, 1, 2)
                .to(device)
            )
            action_mask = create_breakout_action_mask()

            return {"observations": torch_obs, "action_mask": action_mask}

        # -- Create test env
        env = AsyncVectorEnv([breakout_env for _ in range(config.NUM_ENVS)])

        # -- Create SummaryWriter
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = "logs/gradient_tape/" + current_time + "/train"
        summary_writer = SummaryWriter(train_log_dir)

        # Create buffer
        global_buffer = GlobalBuffer()
        buffers = [ReplayBuffer(global_buffer) for _ in range(config.NUM_ENVS)]

        # -- Create Model
        network = AtariMuZero(device=config.DEVICE)
        gumbel = GumbelMuZero(network, model_config)

        # -- Load from timestep
        time_step = config.STARTING_EPISODE
        network.tft_load_model(episode=time_step)

        # -- Create Trainer
        trainer = Trainer(network, summary_writer)

        while True:
            self.rollout(env, preprocess_obs, global_buffer, gumbel, buffers)

            while global_buffer.available_batch():
                batch = global_buffer.sample_gumbel_batch()
                trainer.train_network(batch, time_step)
                time_step += 1

                if time_step % 100 == 0:
                    network.tft_save_model(time_step, trainer.optimizer)
