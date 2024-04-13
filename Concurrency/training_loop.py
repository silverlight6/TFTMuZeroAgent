import time
import datetime
import config
from torch.utils.tensorboard import SummaryWriter
from Models.MuZero_torch_trainer import Trainer


"""
Description - 
    Wrapper function that encapsulates the data transfer from the global buffer to the trainer. Async to sync with ray.
Inputs      - 
    global_agent
        the global model that is sent to the trainer to update. 
    global_buffer
            A buffer that all the individual game buffers send their information to.
"""
class TrainingLoop:
    def __init__(self, global_agent, global_buffer):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        self.summary_writer = SummaryWriter(train_log_dir)
        self.trainer = Trainer(global_agent, self.summary_writer)
        self.batch_size = config.BATCH_SIZE
        self.global_buffer = global_buffer
        self.ckpt_time = time.time_ns()
        self.checkpoint_steps = config.CHECKPOINT_STEPS
        self.champ_decider = config.CHAMP_DECIDER

    """
    Description - 
    Inputs      - 
        global_agent
            the global model that is sent to the trainer to update. 
        storage
            An object that stores global information like the weights of the global model and current training progress
        train_step 
            Current episode that is used for logging and labelling the checkpoints.
    """
    async def loop(self, global_agent, storage, train_step):
        while True:
            if await self.global_buffer.available_batch():
                gameplay_experience_batch = await self.global_buffer.sample_batch()

                self.trainer.train_network(gameplay_experience_batch, train_step)
                # Leaving these comments here because they are benchmarking times for debugging.
                # print("One round in the trainer took {} time".format(time.time_ns() - self.ckpt_time))
                # self.ckpt_time = time.time_ns()
                storage.set_trainer_busy.remote(False)
                storage.set_target_model.remote(global_agent.get_weights())
                train_step += 1

                # Because the champ decider produces 125 samples per game whereas the standard trainer produces
                # closer to 3000, setting the checkpoints to be produced more rapidly.
                if (train_step % self.checkpoint_steps == 0) or \
                        (self.champ_decider and train_step % self.checkpoint_steps % 10 == 0):
                    storage.store_checkpoint.remote(train_step)
                    global_agent.tft_save_model(train_step)
