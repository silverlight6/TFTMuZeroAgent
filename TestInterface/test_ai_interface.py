import datetime

from torch.utils.tensorboard import SummaryWriter

import config
from Core.TorchModels.MuZero_position_torch_agent import MuZero_Position_Network as PositionNetwork
from Core.TorchModels.MuZero_torch_agent import MuZeroNetwork as TFTNetwork
from Core.Trainers.MuZero_position_trainer import Trainer as Position_Trainer
from Core.Trainers.MuZero_torch_trainer import Trainer
from Simulator.observation.vector.observation import ObservationVector
from Simulator.tft_config import TFTConfig
from Simulator.tft_simulator import parallel_env
from Simulator.tft_vector_simulator import TFT_Vector_Pos_Simulator
from TestInterface.test_data_worker import DataWorker
from TestInterface.test_global_buffer import GlobalBuffer
from TestInterface.test_replay_wrapper import BufferWrapper


class AIInterface:

    def __init__(self):
        ...

    def train_model(self, starting_train_step=0):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        train_step = starting_train_step
        model_config = config.ModelConfig()

        global_buffer = GlobalBuffer()

        tftConfig = TFTConfig(observation_class=ObservationVector)
        env = parallel_env(tftConfig)
        data_workers = DataWorker(0, model_config)
        global_agent = TFTNetwork(model_config)
        global_agent.tft_load_model(train_step)

        train_summary_writer = SummaryWriter(train_log_dir)
        trainer = Trainer(global_agent, train_summary_writer)

        while True:
            weights = global_agent.get_weights()
            buffers = BufferWrapper(global_buffer)
            data_workers.collect_gameplay_experience(env, buffers, weights)

            while global_buffer.available_batch():
                gameplay_experience_batch = global_buffer.sample_batch()
                trainer.train_network(gameplay_experience_batch, train_step)
                train_step += 1
                if train_step % 100 == 0:
                    global_agent.tft_save_model(train_step, trainer.optimizer)

    def train_position_model(self, starting_train_step):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        model_config = config.ModelConfig()

        global_buffer = GlobalBuffer()
        train_step = config.STARTING_EPISODE

        env = TFT_Vector_Pos_Simulator(num_envs=config.NUM_ENVS)
        data_workers = DataWorker(0, model_config)
        global_agent = PositionNetwork(model_config).to(config.DEVICE)
        global_agent.tft_load_model(train_step)

        train_summary_writer = SummaryWriter(train_log_dir)
        trainer = Position_Trainer(global_agent, train_summary_writer)

        while True:
            weights = global_agent.get_weights()
            buffers = BufferWrapper(global_buffer)
            average_reward = data_workers.collect_position_experience(env, buffers, weights)
            train_summary_writer.add_scalar("episode_info/reward", average_reward, train_step)

            while global_buffer.available_batch():
                gameplay_experience_batch = global_buffer.sample_position_batch()
                trainer.train_network(gameplay_experience_batch, train_step)
                train_step += 1
                if train_step % 100 == 0:
                    global_agent.tft_save_model(train_step, trainer.optimizer)

    def representation_testing(self):
        import torch
        import torch.nn as nn
        import torch.optim as optim
        import numpy as np
        from Core.TorchModels.Representations.simple_classifier import TFTNetworkTwoClass as TFTNetwork
        from Simulator.batch_generator import BatchGenerator
        from sklearn.metrics import accuracy_score

        tftConfig = TFTConfig()
        from Simulator.observation.token.basic_observation import ObservationToken
        tftConfig.observation_class = ObservationToken
        batch_generator = BatchGenerator(tftConfig)

        model = TFTNetwork(model_config=config.ModelConfig())

        criterion = nn.CrossEntropyLoss()
        # Learning rates of 0.05 and 0.01 both fail to converge
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        num_epochs = 500

        for epoch in range(num_epochs):
            model.train()

            observation, labels = batch_generator.generate_batch(batch_size=config.BATCH_SIZE)
            # print(f"observation[0][4:9] {observation['traits'][0][4:9]} with shape {observation['traits'].shape}")
            X_train_board = torch.from_numpy(observation["board"]).to(config.DEVICE)
            X_train_trait = torch.from_numpy(observation["traits"]).to(config.DEVICE)
            # X_train_item = torch.from_numpy(observation["items"]).to(config.DEVICE)
            # X_train_bench = torch.from_numpy(observation["bench"]).to(config.DEVICE)
            # X_train_shop = torch.from_numpy(observation["shop"]).to(config.DEVICE)
            # X_train_scalars = torch.from_numpy(observation["scalars"]).to(config.DEVICE)
            # X_train_emb_scalars = torch.from_numpy(observation["emb_scalars"]).to(config.DEVICE)

            labels_traits = [label[0] for label in labels]
            labels_traits = [torch.tensor([np.argmax(label[i]) for label in labels_traits])
                             for i in range(len(config.TEAM_TIERS_VECTOR))]

            labels_champ = [label[1] for label in labels]
            labels_champ = [torch.tensor([np.argmax(label[i]) for label in labels_champ])
                            for i in range(len(config.CHAMPION_LIST_DIM))]

            # labels_shop = [label[2] for label in labels]
            # labels_shop = [torch.tensor([np.argmax(label[i]) for label in labels_shop])
            #                for i in range(len([63, 63, 63, 63, 63]))]
            #
            # labels_item = [label[3] for label in labels]
            # labels_item = [torch.tensor([np.argmax(label[i]) for label in labels_item])
            #                for i in range(len([60 for _ in range(10)]))]
            #
            # labels_scalar = [label[4] for label in labels]
            # labels_scalar = [torch.tensor([np.argmax(label[i]) for label in labels_scalar])
            #                  for i in range(len([100 for _ in range(3)]))]

            # trait_output, champ_output, shop_output, item_output, scalar_output = \
            #     model(X_train_trait, X_train_board, X_train_item, X_train_bench, X_train_shop, X_train_scalars,
            #           X_train_emb_scalars)

            trait_output, champ_output = model(X_train_trait, X_train_board)

            loss = 0
            for output, target in zip(trait_output, labels_traits):
                loss += criterion(output, torch.tensor(target, dtype=torch.long).to(config.DEVICE))

            for output, target in zip(champ_output, labels_champ):
                loss += criterion(output, torch.tensor(target, dtype=torch.long).to(config.DEVICE))

            # for output, target in zip(item_output, labels_item):
            #     loss += criterion(output, torch.tensor(target, dtype=torch.long).to(config.DEVICE))
            #
            # for output, target in zip(shop_output, labels_shop):
            #     loss += criterion(output, torch.tensor(target, dtype=torch.long).to(config.DEVICE))

            # for output, target in zip(scalar_output, labels_scalar):
            #     loss += criterion(output, torch.tensor(target, dtype=torch.long).to(config.DEVICE))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print epoch loss
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            observation, labels = batch_generator.generate_batch(batch_size=config.BATCH_SIZE)
            X_test_board = torch.from_numpy(observation["board"]).to(config.DEVICE)
            X_test_trait = torch.from_numpy(observation["traits"]).to(config.DEVICE)
            # X_test_item = torch.from_numpy(observation["items"]).to(config.DEVICE)
            # X_test_bench = torch.from_numpy(observation["bench"]).to(config.DEVICE)
            # X_test_shop = torch.from_numpy(observation["shop"]).to(config.DEVICE)
            # X_test_scalars = torch.from_numpy(observation["scalars"]).to(config.DEVICE)
            # X_test_emb_scalars = torch.from_numpy(observation["emb_scalars"]).to(config.DEVICE)

            labels_traits = [label[0] for label in labels]
            labels_traits = [torch.tensor([np.argmax(label[i]) for label in labels_traits])
                             for i in range(len(config.TEAM_TIERS_VECTOR))]

            labels_champ = [label[1] for label in labels]
            labels_champ = [torch.tensor([np.argmax(label[i]) for label in labels_champ])
                            for i in range(len(config.CHAMPION_LIST_DIM))]

            # labels_shop = [label[2] for label in labels]
            # labels_shop = [torch.tensor([np.argmax(label[i]) for label in labels_shop])
            #                for i in range(len([63, 63, 63, 63, 63]))]
            #
            # labels_item = [label[3] for label in labels]
            # labels_item = [torch.tensor([np.argmax(label[i]) for label in labels_item])
            #                for i in range(len([60 for _ in range(10)]))]
            #
            # labels_scalar = [label[4] for label in labels]
            # labels_scalar = [torch.tensor([np.argmax(label[i]) for label in labels_scalar])
            #                  for i in range(len([100 for _ in range(3)]))]

            # trait_output, champ_output, shop_output, item_output, scalar_output = \
            #     model(X_test_trait, X_test_board, X_test_item, X_test_bench, X_test_shop, X_test_scalars,
            #           X_test_emb_scalars)
            trait_output, champ_output = model(X_test_trait, X_test_board)
            # print(f"trait_output[0] {shop_output[0]}, champ_output[0] {shop_output[0]}")

            y_pred_trait = [torch.argmax(output, axis=1).detach().cpu() for output in trait_output]
            y_pred_champ = [torch.argmax(output, axis=1).detach().cpu() for output in champ_output]
            # y_pred_shop = [torch.argmax(output, axis=1).detach().cpu() for output in shop_output]
            # y_pred_item = [torch.argmax(output, axis=1).detach().cpu() for output in item_output]
            # y_pred_scalar = [torch.argmax(output, axis=1).detach().cpu() for output in scalar_output]

            trait_acc = []
            for y_pred, target in zip(y_pred_trait, labels_traits):
                trait_acc.append(accuracy_score(target.numpy(), y_pred.numpy()))

            comp_acc = []
            for y_pred, target in zip(y_pred_champ, labels_champ):
                comp_acc.append(accuracy_score(target.numpy(), y_pred.numpy()))

            # item_acc = []
            # for y_pred, target in zip(y_pred_items, labels_item):
            #     item_acc.append(accuracy_score(target.numpy(), y_pred.numpy()))
            #
            # shop_acc = []
            # for y_pred, target in zip(y_pred_shop, labels_shop):
            #     shop_acc.append(accuracy_score(target.numpy(), y_pred.numpy()))
            #
            # scalar_acc = []
            # for y_pred, target in zip(y_pred_scalar, labels_scalar):
            #     scalar_acc.append(accuracy_score(target.numpy(), y_pred.numpy()))

            # print(f"y_pred {y_pred_classes}, y_test {labels}")
            # print(f"Test shop Accuracy: {shop_acc}")
            print(f"Test Trait Accuracy: {trait_acc}, Test Comp Accuracy {comp_acc}")
            # print(f"Test Shop Accuracy {shop_acc}Test Item Accuracy: {item_acc}, Test Scalar Accuracy {scalar_acc}")


