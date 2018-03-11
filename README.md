# Reinforcement Learning Toolkit

Reinforcement learning in OpenAI Gym, rllab and DeepMind Control Suite environments.

## Requirements
* Python 3
* Numpy
* imageio
* TensorFlow ~1.4.0
* MPI / mpi4py
* OpenAI Gym

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

## Play back
```sh
CUDA_VISIBLE_DEVICES="" \
PYTHONPATH=.:$PYTHONPATH \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/external_libs/libGL.so \
play.py [-h] [--logdir LOGDIR] [--checkpoint CHECKPOINT]
             [--save-rollouts SAVE_ROLLOUTS] [--save-videos SAVE_VIDEOS]
             [--human-render] [--no-human-render]
             [--num-rollouts NUM_ROLLOUTS]
             [--max-episode-length MAX_EPISODE_LENGTH]
```

### Supporting rllab
rllab is provided as a git submodule which requires additional dependencies that can be installed via
```
pip install cached_property path.py mako
```

### Supporting DeepMind Control Suite
Follow the instructions from the respective [GitHub repository](https://github.com/deepmind/dm_control).
RLToolkit will use the provided submodule as `dm_control` instance.