# Reinforcement Learning Toolkit

Reinforcement learning in OpenAI Gym, rllab and DeepMind Control Suite environments.

## Requirements
* Python 3
* Numpy
* imageio
* TensorFlow ~1.4.0
* MPI / mpi4py
* OpenAI Gym

### Set up submodules
To install the git submodules, run
```sh
git submodule update --init --recursive
```

## Reinforcement Learning
Train for the cartpole swing-up task from DeepMind Control Suite via TRPO:
```sh
export NUM_CPU=8
mpirun -np $NUM_CPU python3 rl.py \
        --num-cpu $NUM_CPU \
        --method trpo \
        --environment dm-cartpole-swingup \
        --num-timesteps 100000
```

### Supporting rllab
rllab is provided as a git submodule which requires additional dependencies that can be installed via
```
pip install cached_property path.py mako
```

### Supporting DeepMind Control Suite
Follow the instructions from the respective [GitHub repository](https://github.com/deepmind/dm_control).
