import time
import config
import datetime
import ray
from torch.utils.tensorboard import SummaryWriter
from Models.MuZero_torch_trainer import Trainer

@ray.remote(num_gpus=1)
class TrainingLoop:
    def __init__(self, global_agent):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        self.summary_writer = SummaryWriter(train_log_dir)
        self.trainer = Trainer(global_agent, self.summary_writer)
        self.batch_size = config.BATCH_SIZE

    def loop(self, global_agent, global_buffer, storage, train_step):
        while True:
            if ray.get(global_buffer.available_batch.remote()):
                gameplay_experience_batch = global_buffer.sample_batch.remote()
                ckpt_time = time.time_ns()
                self.trainer.train_network(gameplay_experience_batch, train_step)
                print("TOTAL_TRAINER_TIME {}".format(time.time_ns() - ckpt_time))
                storage.set_trainer_busy.remote(False)
                storage.set_target_model.remote(global_agent.get_weights())
                train_step += 1
                if train_step % config.CHECKPOINT_STEPS == 0:
                    storage.store_checkpoint.remote(train_step)
                    global_agent.tft_save_model(train_step)
