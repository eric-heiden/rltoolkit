import sys, os, subprocess

import glfw
from gym import Wrapper
from gym.envs.mujoco import MujocoEnv
from mpi4py import MPI

import gym, imageio
import tensorflow as tf
import numpy as np

from typing import Callable

from mujoco_py import MjRenderContextOffscreen, MjViewer, cymj

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'baselines'))
from baselines.acktr.policies import GaussianMlpPolicy


class BetterRgbRenderingEnv(Wrapper):
    def __init__(self, env, camera_name=None, render_width=512, render_height=512):
        super().__init__(env)
        self._mujoco_env = isinstance(env, MujocoEnv)
        if self._mujoco_env and camera_name is None:
            camera_name = env.sim.model.camera_names[0]
        self.camera_name = camera_name
        self.render_width = render_width
        self.render_height = render_height
        # TODO this still opens a dead window
        self.env.sim.add_render_context(cymj.MjRenderContextOffscreen(self.env.sim, 0))

    def render(self, mode='human'):
        if self._mujoco_env and mode == 'rgb_array':
            data = self.env.sim.render(
                self.render_width,
                self.render_height,
                camera_name=self.camera_name,
                device_id=0,
                mode='offscreen')
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        return self.env.render(mode)


def create_environment(name: str) -> gym.Env:
    ids = name.split('-')
    framework = ids[0].lower()
    env_id = '-'.join(ids[1:])
    if framework == 'dm':
        from envs.deepmind import DMSuiteEnv
        return DMSuiteEnv(env_id)
    elif framework == 'gym':
        return BetterRgbRenderingEnv(gym.make(env_id).env)
    elif framework == 'rllab':
        from envs.rllab import RllabEnv
        return RllabEnv(env_id)

    raise LookupError("Could not find environment \"%s\"." % env_id)


def load_dm_control():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dm_control'))
    print("Setting up MuJoCo bindings for dm_control...")
    DEFAULT_HEADERS_DIR = '~/.mujoco/mjpro150/include'

    # Relative paths to the binding generator script and the output directory.
    AUTOWRAP_PATH = 'dm_control/dm_control/autowrap/autowrap.py'
    MJBINDINGS_DIR = 'dm_control/dm_control/mujoco/wrapper/mjbindings'

    # We specify the header filenames explicitly rather than listing the contents
    # of the `HEADERS_DIR` at runtime, since it will probably contain other stuff
    # (e.g. `glfw.h`).
    HEADER_FILENAMES = [
        'mjdata.h',
        'mjmodel.h',
        'mjrender.h',
        'mjvisualize.h',
        'mjxmacro.h',
        'mujoco.h',
    ]

    # A default value must be assigned to each user option here.
    inplace = 0
    headers_dir = os.path.expanduser(DEFAULT_HEADERS_DIR)

    header_paths = []
    for filename in HEADER_FILENAMES:
        full_path = os.path.join(headers_dir, filename)
        if not os.path.exists(full_path):
            raise IOError('Header file {!r} does not exist.'.format(full_path))
        header_paths.append(full_path)
    header_paths = ' '.join(header_paths)

    cwd = os.path.realpath(os.curdir)
    output_dir = os.path.join(cwd, MJBINDINGS_DIR)
    command = [
        sys.executable or 'python',
        AUTOWRAP_PATH,
        '--header_paths={}'.format(header_paths),
        '--output_dir={}'.format(output_dir)
    ]
    old_environ = os.environ.copy()
    try:
        # Prepend the current directory to $PYTHONPATH so that internal imports
        # in `autowrap` can succeed before we've installed anything.
        new_pythonpath = [cwd]
        if 'PYTHONPATH' in old_environ:
            new_pythonpath.append(old_environ['PYTHONPATH'])
        os.environ['PYTHONPATH'] = ':'.join(new_pythonpath)
        subprocess.check_call(command)
    finally:
        os.environ = old_environ

    # XXX this hack works around bug in dm_control where it cannot find a GL environment
    try:
        from dm_control.render.glfw_renderer import GLFWContext as _GLFWRenderer
    except:
        pass

    try:
        from dm_control.render.glfw_renderer import GLFWContext as _GLFWRenderer
    except:
        pass

    try:
        from dm_control.render.glfw_renderer import GLFWContext as _GLFWRenderer
    except:
        pass


def load_state(fname: str):
    saver = tf.train.Saver()
    saver.restore(tf.get_default_session(), fname)


def save_state(fname: str):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    saver = tf.train.Saver()
    saver.save(tf.get_default_session(), fname)


def _render_frames(env: gym.Env):
    return env.render(mode='rgb_array'),


def create_callback_handler(logger, env_name: str, env: gym.Env, method: str, folder: str,
                            episode_length: int=1000, save_every: int=50,
                            video_width: int=400, video_height: int=400,
                            plot_rewards: bool=True, load_policy: str=None,
                            render_frames: Callable[[gym.Env], np.array]=_render_frames)\
        -> Callable[[dict, dict], None]:

    def callback(locals: dict, _globals: dict):
        if method != "ddpg":
            if load_policy is not None and locals['iters_so_far'] == 0:
                # noinspection PyBroadException
                try:
                    load_state(load_policy)
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
                if method == "ddpg":
                    ac, _ = locals['agent'].pi(ob, apply_noise=False, compute_Q=False)
                elif method == "sql":
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
                             (env_name, method, locals['iters_so_far'])),
                video,
                fps=60)
            env.reset()

            if method != "ddpg":
                save_state(os.path.join(folder, "checkpoints", "%s_%i" %
                                        (env_name, locals['iters_so_far'])))

    return callback


def constfn(val):
    def f(_):
        return val
    return f


def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)
