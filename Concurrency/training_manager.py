import ray
import asyncio
from Concurrency.training_loop import TrainingLoop
from Concurrency.global_buffer import GlobalBuffer
from config import TRAINER_GPU_SIZE


class Empty(Exception):
    pass


class Full(Exception):
    pass


"""
Description - 
    Wrapper object for the _TrainActor. This is the object sent to all other ray objects to interact with.
Inputs      - 
    global_agent
        the global model that is sent to the trainer to update. 
    storage
        An object that stores global information like the weights of the global model and current training progress
"""
class TrainingManager:
    def __init__(self, global_agent, storage):
        self.training_ray_manager = _TrainActor.remote(global_agent, storage)
        self.global_agent = global_agent

    """
    Description - 
        Initiates the ray side of the training loop. .remote on an async works the same as it would a normal ray method.
    Inputs      - 
        storage
            An object that stores global information like the weights of the global model and current training progress
        train_step 
            Current episode that is used for logging and labelling the checkpoints.
    """
    def loop(self, storage, train_step):
        self.training_ray_manager.loop.remote(storage, train_step)

    """
    Description - 
        Outward facing api call to the store_replay_sequence.
    Inputs      - 
        samples 
            All of the samples from one game from one agent.
    """
    def store_replay_sequence(self, samples):
        self.training_ray_manager.store_replay_sequence.remote(samples)

    """
    Description - 
        Outward facing buffer size call.
    Outputs     - 
        Buffer size - int
    """
    def buffer_size(self):
        return ray.get(self.training_ray_manager.buffer_size.remote())


"""
Description - 
    Wrapper object for the core training methods. Initialize the global buffer and the TrainingLoop here.
    Takes num_gpus = 1 because the training loop requires some gpu usage to run the trainer. 
Inputs      - 
    global_agent
        the global model that is sent to the trainer to update. 
    storage
        An object that stores global information like the weights of the global model and current training progress
"""
@ray.remote(num_gpus=TRAINER_GPU_SIZE)
class _TrainActor:
    def __init__(self, global_agent, storage):
        self.global_buffer = GlobalBuffer(storage)
        self.training_loop = TrainingLoop(global_agent, self.global_buffer,
                                          ray.get(storage.get_optimizer_dict.remote()))

    """
    Description - 
        Initiates the asyncio side of the training loop. Await allows it be called by a ray.remote call.
    Inputs      - 
        storage
            An object that stores global information like the weights of the global model and current training progress
        train_step 
            Current episode that is used for logging and labelling the checkpoints.
    """
    async def loop(self, storage, train_step):
        await self.training_loop.loop(storage, train_step)

    """
    Description - 
        Inward facing api call to the store_replay_sequence. Should never hit the timeout error.
    Inputs      - 
        samples 
            All of the samples from one game from one agent.
    """
    async def store_replay_sequence(self, samples):
        try:
            await asyncio.wait_for(self.global_buffer.store_replay_sequence(samples), None)
        except asyncio.TimeoutError:
            raise Full

    """
    Description - 
        Inward facing buffer size call. Access value directly to avoid adding additional method calls to the ray object.
    Outputs     - 
        Buffer size - int
    """
    def buffer_size(self):
        return len(self.global_buffer.gameplay_experiences)
        # return self.global_buffer.gameplay_experiences.size
