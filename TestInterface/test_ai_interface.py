import datetime

import config
from TestInterface.test_global_buffer import GlobalBuffer
from TestInterface.test_data_worker import DataWorker
from Simulator.tft_simulator import parallel_env, TFTConfig
from Simulator.observation.vector.observation import ObservationVector
from TestInterface.test_replay_wrapper import BufferWrapper
from Models.MuZero_torch_agent import MuZeroNetwork as TFTNetwork
from Models.MuZero_torch_trainer import Trainer
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
                    global_agent.tft_save_model(train_step)
    