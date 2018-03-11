#!/usr/bin/env python3

import imageio, os, sys, datetime, pathlib, jsonpickle
import utils
import subprocess
from mpi4py import MPI
import os.path as osp
import gym, gym.spaces, logging

import tensorflow as tf
import numpy as np

sys.path.insert(0, osp.join(osp.dirname(__file__), 'baselines'))
sys.path.insert(0, osp.join(osp.dirname(__file__), 'rllab'))
sys.path.insert(0, osp.join(osp.dirname(__file__), 'softqlearning'))
utils.load_dm_control()

from baselines import bench, logger
from baselines.acktr.policies import GaussianMlpPolicy
from baselines.acktr.value_functions import NeuralNetValueFunction
from baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)
from baselines.ppo1 import pposgd_simple, mlp_policy
from baselines.trpo_mpi import trpo_mpi
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.memory import Memory
from baselines.ddpg.noise import *


def _render_frames(env):
    return env.render(mode='rgb_array'),


class RLToolkit:
    def __init__(self,
                 environment,
                 method,
                 workers=1,
                 master=None,
                 gpu_usage=1.):
        self.workers = workers
        self.master = master
        self.environment = environment
        self.method = method
        self.gpu_usage = min(1., gpu_usage)
        self.folder = None

    @staticmethod
    def load_state(fname):
        saver = tf.train.Saver()
        saver.restore(tf.get_default_session(), fname)

    @staticmethod
    def save_state(fname):
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        saver = tf.train.Saver()
        saver.save(tf.get_default_session(), fname)

    def make_folders(self, logdir='logs', **_):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        folder_name = "{timestamp}-{environment}-{method}".format(
            timestamp=timestamp, environment=self.environment, method=self.method)
        folder_name = os.path.abspath(os.path.join(logdir, folder_name))
        pathlib.Path(folder_name, "videos").mkdir(parents=True, exist_ok=True)
        pathlib.Path(folder_name, "checkpoints").mkdir(
            parents=True, exist_ok=True)
        self.folder = folder_name
        return folder_name, timestamp

    def setup_logging(self, kwargs):
        folder_name, timestamp = self.make_folders(**kwargs)
        logger.configure(dir=folder_name, format_strs=['log', 'stdout'])

        run_json = {
            "time": timestamp,
            "settings": kwargs
        }

        if len(sys.argv) > 0:
            run_json["src_files"] = {
                osp.basename(sys.argv[0]): "".join(open(sys.argv[0], "r"))
            }

        # noinspection PyBroadException
        try:
            import git
            repo = git.Repo(search_parent_directories=True)
            run_json["git"] = {
                "head": repo.head.object.hexsha,
                "branch": str(repo.active_branch),
                "summary": repo.head.object.summary,
                "time": repo.head.object.committed_datetime.strftime("%Y-%m-%d-%H-%M-%S"),
                "author": {
                    "name": repo.head.object.author.name,
                    "email": repo.head.object.author.email
                }
            }
        except:
            print("Could not gather git repo information.", file=sys.stderr)

        with open(os.path.join(folder_name, "run.json"), "w") as f:
            f.write(jsonpickle.encode(run_json))

        return folder_name, timestamp

    def train(self,
              env_fn,
              num_timesteps,
              noise_type,
              layer_norm,
              folder,
              load_policy,
              video_width,
              video_height,
              plot_rewards,
              save_every=50,
              seed=1234,
              episode_length=1000,
              pi_hid_size=150,
              pi_num_hid_layers=3,
              render_frames=_render_frames,
              **kwargs):
        num_cpu = self.workers
        if sys.platform == 'darwin':
            num_cpu //= 2
        config = tf.ConfigProto(
            allow_soft_placement=True,
            intra_op_parallelism_threads=num_cpu,
            inter_op_parallelism_threads=num_cpu)

        if self.gpu_usage is None or self.gpu_usage <= 0.:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        else:
            config.gpu_options.allow_growth = True  # pylint: disable=E1101
            config.gpu_options.per_process_gpu_memory_fraction = self.gpu_usage / self.workers
        tf.Session(config=config).__enter__()

        worker_seed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
        set_global_seeds(worker_seed)

        tf.set_random_seed(worker_seed)
        np.random.seed(worker_seed)

        save_every = max(1, save_every)

        env = env_fn()
        env.seed(worker_seed)

        rank = MPI.COMM_WORLD.Get_rank()
        logger.info('rank {}: seed={}, logdir={}'.format(rank, worker_seed,
                                                         logger.get_dir()))

        def policy_fn(name, ob_space, ac_space):
            return mlp_policy.MlpPolicy(
                name=name,
                ob_space=ob_space,
                ac_space=ac_space,
                hid_size=pi_hid_size,
                num_hid_layers=pi_num_hid_layers)

        # env = bench.Monitor(
        #     env,
        #     logger.get_dir() and osp.join(logger.get_dir(), str(rank)),
        #     allow_early_resets=True)
        gym.logger.setLevel(logging.INFO)

        that = self

        def callback(locals, globals):
            if that.method != "ddpg":
                if load_policy is not None and locals['iters_so_far'] == 0:
                    # noinspection PyBroadException
                    try:
                        self.load_state(load_policy)
                        if MPI.COMM_WORLD.Get_rank() == 0:
                            logger.info("Loaded policy network weights from %s." % load_policy)
                            # save TensorFlow summary (contains at least the graph definition)
                    except:
                        logger.error("Failed to load policy network weights from %s." % load_policy)
                if MPI.COMM_WORLD.Get_rank() == 0 and locals['iters_so_far'] == 0:
                    _ = tf.summary.FileWriter(folder, tf.get_default_graph())
            if MPI.COMM_WORLD.Get_rank() == 0 and locals['iters_so_far'] % save_every == 0:
                print('Saving video and checkpoint for policy at iteration %i...' %
                      locals['iters_so_far'])
                ob = env.reset()
                images = []
                rewards = []
                max_reward = 1.  # if any reward > 1, we have to rescale
                lower_part = video_height // 5
                for i in range(episode_length):
                    if that.method == "ddpg":
                        ac, _ = locals['agent'].pi(ob, apply_noise=False, compute_Q=False)
                    elif that.method == "sql":
                        ac, _ = locals['policy'].get_action(ob)
                    elif isinstance(locals['pi'], GaussianMlpPolicy):
                        ac, _, _ = locals['pi'].act(np.concatenate((ob, ob)))
                    else:
                        ac, _ = locals['pi'].act(False, ob)
                    ob, rew, new, _ = env.step(ac)
                    images.append(render_frames(env))
                    if plot_rewards:
                        rewards.append(rew)
                        max_reward = max(rew, max_reward)
                    if new:
                        break

                orange = np.array([255, 163, 0])
                red = np.array([255, 0, 0])
                video = []
                width_factor = 1. / episode_length * video_width
                for i, imgs in enumerate(images):
                    for img in imgs:
                        img[-lower_part, :10] = orange
                        img[-lower_part, -10:] = orange
                        if episode_length < video_width:
                            p_rew_x = 0
                            for j, r in enumerate(rewards[:i]):
                                rew_x = int(j * width_factor)
                                if r < 0:
                                    img[-1:, p_rew_x:rew_x] = red
                                    img[-1:, p_rew_x:rew_x] = red
                                else:
                                    rew_y = int(r / max_reward * lower_part)
                                    img[-rew_y - 1:, p_rew_x:rew_x] = orange
                                    img[-rew_y - 1:, p_rew_x:rew_x] = orange
                                p_rew_x = rew_x
                        else:
                            for j, r in enumerate(rewards[:i]):
                                rew_x = int(j * width_factor)
                                if r < 0:
                                    img[-1:, rew_x] = red
                                    img[-1:, rew_x] = red
                                else:
                                    rew_y = int(r / max_reward * lower_part)
                                    img[-rew_y - 1:, rew_x] = orange
                                    img[-rew_y - 1:, rew_x] = orange
                    video.append(np.hstack(imgs))

                imageio.mimsave(
                    os.path.join(folder, "videos", "%s_%s_iteration_%i.mp4" %
                                 (that.environment, that.method, locals['iters_so_far'])),
                    video,
                    fps=60)
                env.reset()

                if that.method != "ddpg":
                    that.save_state(os.path.join(that.folder, "checkpoints", "%s_%i" %
                                                 (that.environment, locals['iters_so_far'])))

        if self.method == "ppo":
            pposgd_simple.learn(
                env,
                policy_fn,
                max_timesteps=int(num_timesteps),
                timesteps_per_actorbatch=1024,  # 256
                clip_param=0.2,
                entcoeff=0.01,
                optim_epochs=4,
                optim_stepsize=1e-3,  # 1e-3
                optim_batchsize=64,
                gamma=0.99,
                lam=0.95,
                schedule='linear',  # 'linear'
                callback=callback)
        elif self.method == "trpo":
            trpo_mpi.learn(
                env,
                policy_fn,
                max_timesteps=int(num_timesteps),
                timesteps_per_batch=1024,
                max_kl=0.1,  # 0.01
                cg_iters=10,
                cg_damping=0.1,
                gamma=0.99,
                lam=0.98,
                vf_iters=5,
                vf_stepsize=1e-3,
                callback=callback)
        elif self.method == "acktr":
            from algos.acktr import acktr
            with tf.Session(config=tf.ConfigProto()):
                ob_dim = env.observation_space.shape[0]
                ac_dim = env.action_space.shape[0]
                with tf.variable_scope("vf"):
                    vf = NeuralNetValueFunction(ob_dim, ac_dim)
                with tf.variable_scope("pi"):
                    policy = GaussianMlpPolicy(ob_dim, ac_dim)
                acktr.learn(
                    env,
                    pi=policy,
                    vf=vf,
                    gamma=0.99,
                    lam=0.97,
                    timesteps_per_batch=1024,
                    desired_kl=0.01,  # 0.002
                    num_timesteps=num_timesteps,
                    animate=False,
                    callback=callback)
        elif self.method == "ddpg":
            from algos.ddpg import ddpg
            # Parse noise_type
            action_noise = None
            param_noise = None
            nb_actions = env.action_space.shape[-1]
            for current_noise_type in noise_type.split(','):
                current_noise_type = current_noise_type.strip()
                if current_noise_type == 'none':
                    pass
                elif 'adaptive-param' in current_noise_type:
                    _, stddev = current_noise_type.split('_')
                    from baselines.ddpg.noise import AdaptiveParamNoiseSpec
                    param_noise = AdaptiveParamNoiseSpec(
                        initial_stddev=float(stddev),
                        desired_action_stddev=float(stddev))
                elif 'normal' in current_noise_type:
                    _, stddev = current_noise_type.split('_')
                    from baselines.ddpg.noise import NormalActionNoise
                    action_noise = NormalActionNoise(
                        mu=np.zeros(nb_actions),
                        sigma=float(stddev) * np.ones(nb_actions))
                elif 'ou' in current_noise_type:
                    from baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise
                    _, stddev = current_noise_type.split('_')
                    action_noise = OrnsteinUhlenbeckActionNoise(
                        mu=np.zeros(nb_actions),
                        sigma=float(stddev) * np.ones(nb_actions))
                else:
                    raise RuntimeError(
                        'unknown noise type "{}"'.format(current_noise_type))

            # Configure components.
            memory = Memory(
                limit=int(1e6),
                action_shape=env.action_space.shape,
                observation_shape=env.observation_space.shape)
            critic = Critic(layer_norm=layer_norm)
            actor = Actor(nb_actions, layer_norm=layer_norm)

            ddpg.train(
                env=env,
                eval_env=None,
                param_noise=param_noise,
                render=False,
                render_eval=False,
                action_noise=action_noise,
                actor=actor,
                critic=critic,
                memory=memory,
                callback=callback,
                **kwargs)
        elif self.method == "sql":
            from softqlearning.algorithms import SQL
            from softqlearning.misc.kernel import adaptive_isotropic_gaussian_kernel
            from softqlearning.misc.utils import timestamp
            from softqlearning.replay_buffers import SimpleReplayBuffer
            from softqlearning.value_functions import NNQFunction
            from softqlearning.policies import StochasticNNPolicy

            from rllab.envs.gym_env import GymEnv

            env = GymEnv(env)

            variant = {
                'seed': [1, 2, 3],
                'policy_lr': 3E-4,
                'qf_lr': 3E-4,
                'discount': 0.99,
                'layer_size': 128,
                'batch_size': 128,
                'max_pool_size': 1E6,
                'n_train_repeat': 1,
                'epoch_length': 1000,
                'snapshot_mode': 'last',
                'snapshot_gap': 100,
            }

            pool = SimpleReplayBuffer(
                env_spec=env.spec,
                max_replay_buffer_size=variant['max_pool_size'],
            )

            base_kwargs = dict(
                min_pool_size=episode_length,
                epoch_length=episode_length,
                n_epochs=num_timesteps,
                max_path_length=episode_length,
                batch_size=variant['batch_size'],
                n_train_repeat=variant['n_train_repeat'],
                eval_render=False,
                eval_n_episodes=1,
                iter_callback=callback
            )

            qf = NNQFunction(
                env_spec=env.spec,
                hidden_layer_sizes=tuple([pi_hid_size] * pi_num_hid_layers),
            )

            pi_layers = tuple([pi_hid_size] * pi_num_hid_layers)
            policy = StochasticNNPolicy(env_spec=env.spec, hidden_layer_sizes=pi_layers)

            algorithm = SQL(
                base_kwargs=base_kwargs,
                env=env,
                pool=pool,
                qf=qf,
                policy=policy,
                kernel_fn=adaptive_isotropic_gaussian_kernel,
                kernel_n_particles=32,
                kernel_update_ratio=0.5,
                value_n_particles=16,
                td_target_update_interval=1000,
                qf_lr=variant['qf_lr'],
                policy_lr=variant['policy_lr'],
                discount=variant['discount'],
                reward_scale=1,
                save_full_state=False,
            )

            algorithm.train()
        else:
            print('ERROR: Invalid "method" argument provided.', file=sys.stderr)
        env.close()


def main(**kwargs):
    rank = MPI.COMM_WORLD.Get_rank()
    rl = RLToolkit(kwargs['environment'], kwargs['method'])
    if rank == 0:
        folder_name, _ = rl.setup_logging(kwargs)
    else:
        logger.configure(format_strs=[])
        folder_name = None

    def env_fn():
        return utils.create_environment(kwargs['environment'])

    folder_name = MPI.COMM_WORLD.bcast(folder_name, root=0)
    rl.train(env_fn=env_fn, folder=folder_name, **kwargs)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    parser.add_argument(
        '--num-cpu', help='number of CPUs to use', type=int, default=4)
    parser.add_argument(
        '--method',
        help='reinforcement learning algorithm to use (ppo/trpo/ddpg/acktr/sql)',
        type=str,
        default='ppo')
    parser.add_argument(
        '--environment',
        help='environment ID prefixed by framework, e.g. dm-cartpole-swingup, gym-CartPole-v0, rllab-cartpole',
        type=str,
        default='dm-cartpole-swingup')
    # gym-Hopper-v2
    # default='rllab-humanoid')

    parser.add_argument(
        '--logdir',
        help='folder where the logs will be stored',
        type=str,
        default='logs')

    # DDPG settings
    boolean_flag(parser, 'normalize-returns', default=False)
    boolean_flag(parser, 'normalize-observations', default=True)
    boolean_flag(parser, 'layer-norm', default=True)
    parser.add_argument('--critic-l2-reg', type=float, default=1e-2)
    parser.add_argument('--batch-size', type=int, default=64)  # per MPI worker
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    boolean_flag(parser, 'popart', default=False)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--reward-scale', type=float, default=1.)
    parser.add_argument('--clip-norm', type=float, default=None)
    parser.add_argument(
        '--nb-epochs', type=int,
        default=500)  # with default settings, perform 1M steps total
    parser.add_argument('--nb-epoch-cycles', type=int, default=20)
    parser.add_argument(
        '--nb-train-steps', type=int,
        default=50)  # per epoch cycle and MPI worker
    parser.add_argument(
        '--nb-eval-steps', type=int,
        default=100)  # per epoch cycle and MPI worker
    parser.add_argument(
        '--nb-rollout-steps', type=int,
        default=100)  # per epoch cycle and MPI worker
    parser.add_argument(
        '--noise-type', type=str, default='adaptive-param_0.2'
    )  # choices are adaptive-param_xx, ou_xx, normal_xx, none

    # resolution of rendered videos
    parser.add_argument('--video-width', type=int, default=400)
    parser.add_argument('--video-height', type=int, default=400)
    boolean_flag(parser, 'plot-rewards', default=True)
    parser.add_argument('--render-camera', type=int, default=1)
    boolean_flag(parser, 'deterministic-reset', default=False)

    # load existing policy network weights
    parser.add_argument('--load-policy', type=str, default=None)

    args = parser.parse_args()
    main(**vars(args))
