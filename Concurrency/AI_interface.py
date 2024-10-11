import random
import time
import ray
import os
import torch
import config

from Concurrency.storage import Storage
from Simulator.tft_simulator import parallel_env, TFTConfig
from Simulator.observation.token.basic_observation import ObservationToken
from Models.replay_buffer_wrapper import BufferWrapper
from Models.MuZero_torch_agent import MuZeroNetwork as TFTNetwork
from Models.Muzero_default_agent import MuZeroDefaultNetwork as DefaultNetwork
from Concurrency.data_worker import DataWorker
from Concurrency.training_manager import TrainingManager
from Concurrency.queue_storage import QueueStorage



"""
Description - Highest level class for concurrent training. Called from main.py
"""
class AIInterface:

    def __init__(self):
        ...

    '''
    Description - Global train model method. This is what gets called from main.
    Inputs - starting_train_step: int
                Checkpoint number to load. If 0, a fresh model will be created.
    '''
    def train_torch_model(self) -> None:
        gpus = torch.cuda.device_count()
        with ray.init(num_gpus=gpus, num_cpus=config.NUM_CPUS, namespace="TFT_AI"):
            train_step = config.STARTING_EPISODE

            workers = []
            model_config = config.ModelConfig()
            data_workers = [DataWorker.remote(rank, model_config) for rank in range(config.CONCURRENT_GAMES)]
            storage = Storage.remote(train_step)
            if config.CHAMP_DECIDER:
                global_agent = DefaultNetwork(model_config)
            else:
                global_agent = TFTNetwork(model_config)

            global_agent_weights = ray.get(storage.get_target_model.remote())
            global_agent.set_weights(global_agent_weights)
            global_agent.to(config.DEVICE)

            training_manager = TrainingManager(global_agent, storage)

            # Keeping this line commented because this tells us the number of parameters that our current model has.
            # total_params = sum(p.numel() for p in global_agent.parameters())

            tftConfig = TFTConfig(observation_class=ObservationToken)
            env = parallel_env(tftConfig)

            buffers = [BufferWrapper.remote()
                       for _ in range(config.CONCURRENT_GAMES)]

            weights = ray.get(storage.get_target_model.remote())

            for i, worker in enumerate(data_workers):
                workers.append(worker.collect_gameplay_experience.remote(env, buffers[i], training_manager,
                                                                         storage, weights))
                time.sleep(0.5)

            training_manager.loop(storage, train_step)

            # This may be able to be ray.wait(workers). Here so we can keep all processes alive.
            # ray.get(storage)
            ray.get(workers)

    """
    I'll write in here what I want to do with this.
    I need to start by creating an environment where I can get random positions. 
    """
    def representation_testing(self):
        import datetime
        from Models.Representations.representation_model import RepresentationTesting
        from Models.Representations.representation_trainer import RepresentationTrainer
        from torch.utils.tensorboard import SummaryWriter
        train_step = config.STARTING_EPISODE

        model_config = config.ModelConfig()

        global_agent = RepresentationTesting(model_config)
        global_agent.to(config.DEVICE)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        summary_writer = SummaryWriter(train_log_dir)

        rep_trainer = RepresentationTrainer(global_agent, summary_writer)

        while True:
            rep_trainer.train_network(train_step)
            train_step += 1

        # Keeping this line commented because this tells us the number of parameters that our current model has.
        # total_params = sum(p.numel() for p in global_agent.parameters())

    def representation_evauation(self):
        from Models.Representations.representation_model import RepresentationTesting
        from Evaluator.representation_evaluator import RepresentationEvaluator
        gpus = torch.cuda.device_count()
        with ray.init(num_gpus=gpus, num_cpus=config.NUM_CPUS, namespace="TFT_AI"):
            train_step = config.STARTING_EPISODE

            model_config = config.ModelConfig()

            storage = Storage.remote(train_step)

            global_agent = RepresentationTesting(model_config)
            global_agent_weights = ray.get(storage.get_target_model.remote())
            global_agent.set_weights(global_agent_weights)
            global_agent.to(config.DEVICE)

            evaluator = RepresentationEvaluator(global_agent)
            evaluator.evaluate()
            ray.wait()

    # # # Commenting out until the position model is ready to be inserted again.
    # def train_guide_model(self) -> None:
    #     gpus = torch.cuda.device_count()
    #     with ray.init(num_gpus=gpus, num_cpus=config.NUM_CPUS, namespace="TFT_AI"):
    #         train_step = config.STARTING_EPISODE
    #
    #         workers = []
    #         modelConfig = config.ModelConfig()
    #         data_workers = [DataWorker.remote(rank, modelConfig) for rank in range(config.CONCURRENT_GAMES)]
    #         storage = Storage.remote(train_step)
    #
    #         if config.CHAMP_DECIDER:
    #             global_agent = DefaultNetwork(modelConfig)
    #         else:
    #             global_agent = TFTNetwork(modelConfig)
    #
    #         global_agent_weights = ray.get(storage.get_target_model.remote())
    #         global_agent.set_weights(global_agent_weights)
    #         global_agent.to(config.DEVICE)
    #
    #         training_manager = TrainingManager(global_agent, storage)
    #
    #         # Keeping this line commented because this tells us the number of parameters that our current model has.
    #         # total_params = sum(p.numel() for p in global_agent.parameters())
    #
    #         tftConfig = TFTConfig(observation_class=ObservationVector)
    #         env = parallel_env(tftConfig)
    #
    #         buffers = [BufferWrapper.remote()
    #                    for _ in range(config.CONCURRENT_GAMES)]
    #         positioning_storage = QueueStorage(name="position")
    #         item_storage = QueueStorage(name="item")

            # ppo_position_model = PPO_Position_Model.remote(positioning_storage)
            # ppo_item_model = PPO_Item_Model.remote(item_storage)

            # weights = ray.get(storage.get_target_model.remote())
            #
            # for i, worker in enumerate(data_workers):
            #     workers.append(worker.collect_default_experience.remote(env, buffers[i], training_manager, storage,
            #                                                             weights, item_storage, positioning_storage,
            #                                                             ppo_position_model, ppo_item_model))
            #     time.sleep(0.5)
            #
            # training_manager.loop(storage, train_step)
            #
            # # Tests for the position and item environment
            # # test_envs = DataWorker.remote(0)
            # # workers.append(test_envs.test_position_item_simulators.remote(positioning_storage, item_storage))
            #
            # ppo_item_model.train_item_model.remote()
            # ppo_position_model.train_position_model.remote()
            #
            # # This may be able to be ray.wait(workers). Here so we can keep all processes alive.
            # # ray.get(storage)
            # ray.get(workers)

    def position_ppo_testing(self):
        import torch.nn as nn
        import torch.optim as optim
        import numpy as np
        from torch.utils.tensorboard import SummaryWriter

        from Simulator.tft_vector_simulator import TFT_Vector_Pos_Simulator, list_to_dict
        from Models.PositionModels.action_mask_model import TorchPositionModel
        from Models.utils import convert_to_torch_tensor

        ppo_config = config.PPOConfig()

        run_name = f"{ppo_config.EXP_NAME}__{int(time.time())}"

        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(ppo_config).items()])),
        )

        # TRY NOT TO MODIFY: seeding
        random.seed(8)
        np.random.seed(8)
        torch.manual_seed(8)
        # torch.backends.cudnn.deterministic = True

        device = torch.device(config.DEVICE)

        # env setup
        envs = [TFT_Vector_Pos_Simulator.remote(num_envs=ppo_config.NUM_ENVS) for _ in range(ppo_config.NUM_STEPS)]

        model_config = config.ModelConfig()
        agent = TorchPositionModel(model_config).to(device)

        optimizer = optim.Adam(agent.parameters(), lr=ppo_config.LEARNING_RATE, eps=1e-5)

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        reset_env = []
        for env in envs:
            reset_env.append(ray.get(env.vector_reset.remote())[0])

        obs = convert_to_torch_tensor(x=list_to_dict(reset_env), device=config.DEVICE)
        num_updates = ppo_config.TOTAL_TIMESTEPS // ppo_config.BATCH_SIZE
        kl_coef = ppo_config.KL_COEF

        for update in range(1, num_updates + 1):
            # Annealing the rate if instructed to do so.
            if ppo_config.ANNEAL_LR:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * ppo_config.LEARNING_RATE
                optimizer.param_groups[0]["lr"] = lrnow

            global_step += ppo_config.NUM_STEPS * ppo_config.NUM_ENVS

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent(obs)
                values = value.flatten()
            actions = action.cpu().numpy()

            workers = []
            for j in range(ppo_config.NUM_STEPS):
                workers.append(
                    envs[j].vector_reset_step.remote(actions[j * ppo_config.NUM_ENVS:(j + 1) * ppo_config.NUM_ENVS]))

            obs, reward, done = [], [], []
            for worker in workers:
                local_worker = ray.get(worker)
                obs.append(local_worker[0])
                reward.append(local_worker[1])
            obs = list_to_dict(obs)
            reward = torch.tensor(reward)

            # TRY NOT TO MODIFY: execute the game and log data.
            rewards = torch.tensor(reward).to(device).view(-1)
            obs = convert_to_torch_tensor(obs, config.DEVICE)

            advantages = rewards - values
            loss_per_update = 0

            # Optimizing the policy and value network
            clipfracs = []
            for epoch in range(ppo_config.UPDATE_EPOCHS):
                # get new action probabilities and values
                _, newlogprob, entropy, newvalue = agent(obs, actions)
                # calculate log ratio (element-wise for each action dimension)
                logratio = newlogprob - logprob
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > ppo_config.CLIP_COEF).float().mean().item()]

                # Policy loss
                print(advantages.shape)
                print(ratio.shape)
                print(logratio)
                print(ratio)
                time.sleep(2)
                pg_loss1 = -advantages * ratio
                pg_loss2 = -advantages * torch.clamp(ratio, 1 - ppo_config.CLIP_COEF, 1 + ppo_config.CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if ppo_config.CLIP_VLOSS:
                    print(f"values {newvalue[:64]}, rewards {rewards[:64]}")
                    v_loss_unclipped = (newvalue - rewards) ** 2
                    v_clipped = values + torch.clamp(newvalue - values, -ppo_config.CLIP_COEF, ppo_config.CLIP_COEF)
                    v_loss_clipped = (v_clipped - rewards) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    print(f"values {newvalue[:64]}, rewards {rewards[:64]}")
                    v_loss = 0.5 * ((newvalue - rewards) ** 2).mean()

                if approx_kl > ppo_config.TARGET_KL * (1 + ppo_config.KL_ADJUSTER):
                    kl_coef *= (1 + ppo_config.KL_ADJUSTER)
                elif approx_kl < ppo_config.TARGET_KL / (1 + ppo_config.KL_ADJUSTER):
                    kl_coef *= (1 - ppo_config.KL_ADJUSTER)
                kl_coef = max(min(kl_coef, ppo_config.MAX_KL_COEF), ppo_config.MIN_KL_COEF)

                entropy_loss = entropy.mean()
                loss = pg_loss - ppo_config.ENT_COEF * entropy_loss + ppo_config.VF_COEF * v_loss + kl_coef * approx_kl
                loss_per_update += loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), ppo_config.MAX_GRAD_NORM)
                optimizer.step()

            y_pred, y_true = values.cpu().numpy(), rewards.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", ppo_config.VF_COEF * v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/kl_loss", kl_coef * approx_kl.item(), global_step)
            writer.add_scalar("losses/kl_coef", kl_coef, global_step)
            writer.add_scalar("losses/entropy_loss", ppo_config.ENT_COEF * entropy_loss.item(), global_step)
            writer.add_scalar("losses/total_loss", loss_per_update, global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            writer.add_scalar("charts/mean_reward", torch.mean(rewards).detach().cpu(), global_step)
            print(f"Reward: {torch.mean(rewards).detach().cpu()} with policy loss {pg_loss.item()} "
                  f"value_loss {ppo_config.VF_COEF * v_loss.item()}, kl_loss {kl_coef * approx_kl.item()}, "
                  f"entropy loss {ppo_config.ENT_COEF * entropy_loss.item()} and total_loss {loss_per_update}")
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        envs.close()
        writer.close()

    def position_muzero_testing(self):
        return

    def collect_dummy_data(self) -> None:
        """
        Method used for testing the simulator. It does not call any AI and generates random actions from numpy.
        Tests how fast the simulator is and if there are any bugs that can be caught via multiple runs.
        """
        env = parallel_env()
        while True:
            _ = env.reset()
            terminated = {player_id: False for player_id in env.possible_agents}
            t = time.time_ns()
            while not all(terminated.values()):
                # agent policy that uses the observation and info
                action = {
                    agent: env.action_space(agent).sample()
                    for agent in env.agents
                    if (agent in terminated and not terminated[agent])
                }
                observation_list, rewards, terminated, truncated, info = env.step(action)
            print("A game just finished in time {}".format(time.time_ns() - t))

    def evaluate(self, config) -> None:
        """
        The global side to the evaluator. Creates a set of workers to test a series of agents.
        """
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        # gpus = tf.config.list_physical_devices('GPU')
        ray.init(num_gpus=4, num_cpus=16)
        storage = Storage.remote(0)

        env = parallel_env()

        workers = []
        data_workers = [DataWorker.remote(rank) for rank in range(config.CONCURRENT_GAMES)]
        for i, worker in enumerate(data_workers):
            workers.append(worker.evaluate_agents.remote(env, storage))
            time.sleep(1)

        ray.get(workers)
