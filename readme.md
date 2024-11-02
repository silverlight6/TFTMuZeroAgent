# Teamfight Tactics RL Agent

Teamfight Tactics is an auto-battler game made by Riot.

This is the first attempt at purely artificial intelligence algorithm to play Team Fight Tactics.

This implementation uses a battle simulator of TFT Set 4 based on Avadaa's project.

The player rounds as well as the player, pool, and game round classes were designed for this AI project. All aspects of the game from set 4 minus graphics and sounds are implemented in our simulation. It is set up as a multi-agent petting zoo environment.

# Config
This project has many different modes and models that you can run.
Adjust the .env file to run each specific part of the project.

All of these variables are ture or false, true to run, false if not. Only keep one of the following true at any given point in time.
It is not possible to run multiple at the same time.
If none are true, the sampled muzero model will run.

IMITATION -> Trains a Sampled MuZero style model to play the full game of TFT based on the actions from a pre-built bot that uses hard coded logic to select actions.

CHAMP_DECIDER -> Currently runs a PPO model that learns to do positioning in TFT

REP_TRAINER -> Trains a transformer to take a board position and output what champions are on the board as well as what the current traits are

GUMBEL -> Same as default but uses Gumbel MuZero instead of Sampled MuZero

SINGLE_PLAYER -> Single player environment where your agent (using Gumbel MuZero) builds a board to play against a stronger and stronger pre-built opponent

MUZERO_POSITION -> An AlphaZero style agent that learns how to do positioning in TFT

# Recommended Set-up
Python version: ~ Python 3.8

- Create a virtual env to use with the project & activate virtual env
- Install all necessary libraries in codebase

Before starting training, you need to build the c++/cython style external packages. (GCC version 7.5+ is required.)
```
cd core/ctree
bash make.sh

Any and all questions related to this project are welcome at slucoris@gmail.com

Any and all further improvements to this project will be looked at, discussed, and highly likely accepted.

The environment is separated from the model so if someone wants to add an additional model to this environment, they are welcome to do so following the same examples as the current model is set up.

If anyone wants to participate in the project, all are welcome to join the discord at https://discord.gg/cPKwGU7dbU