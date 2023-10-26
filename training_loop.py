import ray
import time
import config
import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from Models.MuZero_torch_trainer import Trainer

# @ray.remote(num_cpus=2, num_gpus=1, scheduling_strategy="SPREAD")
@ray.remote(num_gpus=1)
class TrainingLoop:
    def __init__(self, global_agent):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        self.summary_writer = SummaryWriter(train_log_dir)
        self.trainer = Trainer(global_agent, self.summary_writer)
        self.batch_size = config.BATCH_SIZE

    def loop(self, global_buffer, global_agent, storage, train_step):
        while True:
            if ray.get(global_buffer.available_batch.remote()):
                gameplay_experience_batch = global_buffer.sample_batch.options(num_returns=13).remote()
                # gameplay_experience_generator = global_buffer.sample_batch_generator.options(
                #     num_returns=config.BATCH_SIZE).remote()
                # gameplay_experience_batch = self.preprocess_batch(gameplay_experience_generator)
                ckpt_time = time.time_ns()
                self.trainer.train_network(gameplay_experience_batch, train_step)
                print("TOTAL_TRAINER_TIME {}".format(time.time_ns() - ckpt_time))
                storage.set_trainer_busy.remote(False)
                storage.set_target_model.remote(global_agent.get_weights())
                train_step += 1
                if train_step % config.CHECKPOINT_STEPS == 0:
                    storage.store_checkpoint.remote(train_step)
                    global_agent.tft_save_model(train_step)

    def preprocess_batch(self, batch_generator):
        ckpt_time = time.time_ns()
        batch = ray.get(batch_generator)
        print("Sample batch took {} time".format(time.time_ns() - ckpt_time))
        print(len(batch))
        batch[:][0] = self.reshape_observation(batch[:][0])
        return batch

    def reshape_observation(self, obs_batch):
        obs_reshaped = {}
        print(len(obs_batch))
        for obs in obs_batch:
            # print(obs)
            for key in obs:
                if key not in obs_reshaped:
                    obs_reshaped[key] = []
                obs_reshaped[key].append(obs[key])

        for key in obs_reshaped:
            obs_reshaped[key] = np.stack(obs_reshaped[key], axis=0)

        return obs_reshaped