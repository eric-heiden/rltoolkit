from collections import namedtuple

import gym, sys, traceback
import numpy as np
from dm_control import suite
from dm_control.mujoco.wrapper import mjbindings
from gym.spaces import Box


def _stop_criterion(step, env):
    return step.last()


class DMSuiteEnv(gym.Env):
    def __init__(self,
                 id="cartpole-swingup",
                 visualize_reward=True,
                 deterministic_reset=False,
                 render_camera=1,
                 render_width=400,
                 render_height=400,
                 dm_env=None,
                 stop_criterion=_stop_criterion):
        id = id.split('-')
        if len(id) == 1:
            domain_name, task_name = id[0], "default"
        else:
            domain_name, task_name = id[0], '-'.join(id[1:])
        id = "%s-%s" % (domain_name, task_name)
        if dm_env is None:
            self.dm_env = suite.load(
                domain_name=domain_name,
                task_name=task_name,
                visualize_reward=visualize_reward)
        else:
            self.dm_env = dm_env
        action_spec = self.dm_env.action_spec()
        self.action_space = Box(
            low=action_spec.minimum[0],
            high=action_spec.maximum[0],
            shape=action_spec.shape,
            dtype=np.float32)
        self.deterministic_reset = deterministic_reset

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dm_env.physics.timestep()))
        }
        self.render_camera = render_camera
        self.render_width = render_width
        self.render_height = render_height

        try:
            ob_spec = self.dm_env.task.observation_spec(self.dm_env.physics)
            self.observation_space = gym.spaces.Box(
                low=ob_spec.minimum[0],
                high=ob_spec.maximum[0],
                shape=ob_spec.shape,
                dtype=np.float32)
        except NotImplementedError:
            print("Could not retrieve observation spec, using +/- infinity.",
                  file=sys.stderr)
            # sample observation and set range to [-10, 10]
            # ob = self.dm_env.task.get_observation(self.dm_env.physics)
            # # ob is an OrderedDict, iterate over all entries to determine overall flattened ob dim
            # ob_dimension = 0
            # for entry in ob.values():
            #     ob_dimension += len(entry.flatten())
            ob = self.observe()
            ob_dimension = len(ob)
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(ob_dimension, ),
                dtype=np.float32)
        self.reward_range = (0, 1)
        print('Initialized %s: %s.' % (domain_name, task_name))
        print('\tobservation space: %s (min: %.2f, max: %.2f)' %
              (str(self.observation_space.shape),
               self.observation_space.low[0], self.observation_space.high[0]))
        print('\taction space: %s (min: %.2f, max: %.2f)' %
              (str(self.action_space.shape), self.action_space.low[0],
               self.action_space.high[0]))

        env_spec = namedtuple('env_spec',
                              ['id', 'timestep_limit', 'observation_space', 'action_space'])
        self._spec = env_spec(id=id,
                              timestep_limit=1000,
                              observation_space=self.observation_space,
                              action_space=self.action_space)
        self.stop_criterion = stop_criterion

        self._ep_length = 0

    def step(self, action):
        # noinspection PyBroadException
        try:
            step = self.dm_env.step(action)
            done = self.stop_criterion(step, self)
            reward = step.reward
            if reward is None:
                reward = 0
        except:
            # could only be dm_control.rl.control.PhysicsError?
            # reset environment for bad controls
            print(traceback.format_exc(), file=sys.stderr)
            self.dm_env.reset()
            done = True
            reward = 0

        if done:
            self._ep_length = 0
        else:
            self._ep_length += 1

        ob = self.observe()
        return ob, reward, done, {'episode': {'r': reward, 'l': self._ep_length}}

    def reset(self):
        self.dm_env.reset()
        self.needs_reset = False
        self._ep_length = 0

        if self.deterministic_reset:
            hinge = mjbindings.enums.mjtJoint.mjJNT_HINGE
            slide = mjbindings.enums.mjtJoint.mjJNT_SLIDE
            ball = mjbindings.enums.mjtJoint.mjJNT_BALL
            free = mjbindings.enums.mjtJoint.mjJNT_FREE

            physics = self.dm_env.physics

            for joint_id in range(physics.model.njnt):
                joint_name = physics.model.id2name(joint_id, 'joint')
                joint_type = physics.model.jnt_type[joint_id]
                is_limited = physics.model.jnt_limited[joint_id]
                range_min, range_max = physics.model.jnt_range[joint_id]

                if is_limited:
                    if joint_type == hinge or joint_type == slide:
                        self.dm_env.physics.named.data.qpos[
                            joint_name] = np.mean([
                                range_min, range_max
                            ])  # random.uniform(range_min, range_max)

                    elif joint_type == ball:
                        self.dm_env.physics.named.data.qpos[
                            joint_name] = np.mean([
                                range_min, range_max
                            ])  # random_limited_quaternion(random, range_max)

                else:
                    if joint_type == hinge:
                        self.dm_env.physics.named.data.qpos[joint_name] = 0.  # random.uniform(-np.pi, np.pi)

                    elif joint_type == ball:
                        quat = np.zeros(4)  # random.randn(4)
                        # quat /= np.linalg.norm(quat)
                        self.dm_env.physics.named.data.qpos[joint_name] = quat

                    elif joint_type == free:
                        quat = np.zeros(4)  # random.rand(4)
                        # quat /= np.linalg.norm(quat)
                        self.dm_env.physics.named.data.qpos[joint_name][3:] = quat
            self.dm_env.physics.after_reset()

        return self.observe()

    def seed(self, seed=None):
        self.dm_env.random = np.random.RandomState(seed)
        return [seed]

    def observe(self):
        ob = self.dm_env.task.get_observation(self.dm_env.physics)
        ob = np.concatenate(list(v.flatten() for v in ob.values()))
        return ob

    def render(self, mode='rgb_array', close=False):
        if mode == 'human':
            # there is no better way than matplotlib right now (which is extremely slow):
            # https://github.com/deepmind/dm_control/issues/4
            import matplotlib.pyplot as plt
            img = self.dm_env.physics.render(
                self.render_width,
                self.render_height,
                camera_id=self.render_camera)
            plt.imshow(img)
            plt.pause(self.dm_env.physics.timestep())
            plt.draw()
        elif mode == 'rgb_array':
            return self.dm_env.physics.render(
                self.render_width,
                self.render_height,
                camera_id=self.render_camera)
        else:
            raise NotImplementedError('Render mode %s not implemented for the DM Control Suite environment.' % mode)
