# Teamfight Tactics RL Agent

Teamfight Tactics is an auto chess game made by Riot.

To my knowledge, this is the first attempt at purely artificial intelligence algorithm to play Team Fight Tactics.

This implementation uses a simulation of TFT Set 4 based on Avadaa's project.

The player rounds as well as the player, pool, and game round classes were designed for this AI project. All aspects of the game from set 4 minus graphics and sounds are implemented in our simulation. It is set up as a multi-agent petting zoo environment.

The reinforcement learning algorithm currently in use is MuZero based on google's implementation but adjusted for use here.
Many features in google's implementation were removed and I adjusted the tree to allow for multiple players to take actions at the same time to support batching.

Any and all questions related to this project are welcome at slucoris@gmail.com

Any and all further improvements to this project will be looked at, discussed, and highly likely accepted.

The environment is separated from the model so if someone wants to add an additional model to this environment, they are welcome to do so following the same examples as the current model is set up.

If anyone wants to participate in the project, all are welcome to join the discord at https://discord.gg/cPKwGU7dbU

# Recommended Set-up
Python version: ~ Python 3.8

- Create a virtual env to use with the project & activate virtual env
- Install all necessary libraries in codebase

Before starting training, you need to build the c++/cython style external packages. (GCC version 7.5+ is required.)
```
cd core/ctree
bash make.sh
```

If your `core/ctree` directory contains a `build` directory and a `cytree.cpp` file you should be ready to go.

**Run `python main.py` to start training!**

# Potential Issues Running Your First Time

### PyTorch Installation
You may run into a `AssertionError: Torch not compiled with CUDA enabled`
If you have a GPU that supports CUDA make sure Torch is configured to work with it. You can download the latest stable version of PyTorch here:
https://pytorch.org/get-started/locally/

### Out of Memory
If you run into an issue where CUDA runs out of memory, 
`torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 512.00 MiB.`
you may need to lower the batch size of the trainer.
Find the `BATCH_SIZE` config value in `config.py` and try lowering the default value of 1024 by half until you no longer run into this issue.
`BATCH_SIZE = get_int_env("BATCH_SIZE", 1024)`

# Understanding Progress
Currently, while training, the terminal only displays the amount of data in the buffer. You can observe your progress using the tensorboard.
Run `tensorboard --logdir logs/gradient_tape/` in a 2nd terminal and open `http://localhost:6006/` in a browser.

By default, the trainer will make checkpoints of your progress for every 100 steps.
You can change how often it saves with the `CHECKPOINT_STEPS` value in `config.py`.