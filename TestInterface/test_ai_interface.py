import config
import datetime

from Core.TorchModels.MuZero_torch_agent import MuZeroNetwork as TFTNetwork
from Core.Trainers.MuZero_torch_trainer import Trainer
from Core.TorchModels.MuZero_position_torch_agent import MuZero_Position_Network as PositionNetwork
from Core.Trainers.MuZero_position_trainer import Trainer as Position_Trainer
from Simulator.observation.vector.observation import ObservationVector
from Simulator.tft_simulator import parallel_env, TFTConfig
from Simulator.tft_vector_simulator import TFT_Vector_Pos_Simulator
from TestInterface.test_data_worker import DataWorker
from TestInterface.test_global_buffer import GlobalBuffer
from TestInterface.test_replay_wrapper import BufferWrapper
from torch.utils.tensorboard import SummaryWriter

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
