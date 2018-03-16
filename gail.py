#!/usr/bin/env python3
from abc import ABC, abstractmethod
from collections import deque
from typing import Callable, Union, Optional

import os, sys, time

import enum

import imageio
import joblib
from tqdm import tqdm

import utils
import os.path as osp
import gym, gym.spaces

import tensorflow as tf
import numpy as np
import pickle as pkl

from experiment import log_parameters, Experiment

sys.path.insert(0, osp.join(osp.dirname(__file__), 'baselines'))
sys.path.insert(0, osp.join(osp.dirname(__file__), 'rllab'))

from baselines import bench, logger
from baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)
from baselines.common import set_global_seeds
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env import VecEnv
from baselines.common.distributions import make_pdtype
from baselines.common import explained_variance
from baselines.a2c.utils import fc


# TODO merry rllab Policy class with this ABC definition
class Policy(ABC):
    def __init__(self):
        self.vf = None
        self.pd = None
        self.pdtype = None
        self.initial_state = None
        self.obs = None
        # in case of a recurrent policy
        self.masks = None
        self.states = None

    @abstractmethod
    def step(self, observation, *_args, **_kwargs):
        raise NotImplementedError()

    def value(self, observation, *_args, **_kwargs):
        raise NotImplementedError()


PolicyFunction = Callable[[tf.Session, tuple, tuple, int, int, Optional[Union[bool, enum.Enum]], Optional[str]],
                          Policy]
EnvType = Union[gym.Env, VecEnv, VecNormalize, DummyVecEnv]


# Extracts features from a current time step to be used by the discriminator.
def observations_and_actions(_env: EnvType,
                             _policy: Policy,
                             _time_step: int,
                             observation: np.array,
                             action: np.array,
                             _env_reward: float) -> np.array:
    """
    Selects what the discriminator should see at every time step.
    """
    return np.hstack((observation, action))


class Discriminator:
    """
    Discriminator is a binary classifier.
    """

    @log_parameters
    def __init__(self, state_dim, hidden_layers=(20, 20), hidden_activation=tf.tanh,
                 entcoeff=0.001, learning_rate=1e-3, name_prefix=""):
        # map features (e.g. state-action pairs) to classifier scores
        # i.e. log-probabilities of (fake, real)
        assert len(hidden_layers) > 0
        sess = tf.get_default_session()

        self.scope = name_prefix + 'discriminator'

        input_shape = (None,) + state_dim

        def build_graph(input, reuse: bool):
            with tf.variable_scope(self.scope):
                if reuse:
                    tf.get_variable_scope().reuse_variables()
                hidden = input
                for i, size in enumerate(hidden_layers):
                    hidden = tf.layers.dense(hidden, size, hidden_activation, name='hidden%i' % i)
                logit = tf.layers.dense(hidden, 1, tf.identity, name='logit')
            return logit

        self.generator_features = tf.placeholder(tf.float32, input_shape, name="generator_features")
        self.expert_features = tf.placeholder(tf.float32, input_shape, name="expert_features")

        # Build graph
        generator_logits = build_graph(self.generator_features, reuse=False)
        expert_logits = build_graph(self.expert_features, reuse=True)

        # Build accuracy
        generator_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(generator_logits) < 0.5))
        expert_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(expert_logits) > 0.5))

        # Build regression loss
        # let x = logits, z = targets.
        # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=generator_logits,
                                                                 labels=tf.zeros_like(generator_logits))
        generator_loss = tf.reduce_mean(generator_loss)
        expert_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=expert_logits,
                                                              labels=tf.ones_like(expert_logits))
        expert_loss = tf.reduce_mean(expert_loss)

        # Build entropy loss
        logits = tf.concat([generator_logits, expert_logits], 0)
        entropy = tf.reduce_mean(Discriminator.logit_bernoulli_entropy(logits))
        entropy_loss = -entcoeff * entropy
        # Loss + Accuracy terms
        self.losses = [generator_loss, expert_loss, entropy, entropy_loss, generator_acc, expert_acc]
        self.loss_name = ["generator_loss", "expert_loss", "entropy", "entropy_loss", "generator_acc", "expert_acc"]
        self.total_loss = generator_loss + expert_loss # + entropy_loss
        # Build Reward for policy
        self.reward_op = -tf.log(1 - tf.nn.sigmoid(generator_logits) + 1e-8)
        # var_list = self.get_trainable_variables()
        # self.lossandgrad = U.function(
        #     [self.generator_features, self.expert_features],
        #     self.losses + [U.flatgrad(self.total_loss, var_list)])

        params = self.get_trainable_variables()
        grads = tf.gradients(self.total_loss, params)
        grads = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-5)
        _train = trainer.apply_gradients(grads)

        def classify(features: np.array) -> float:
            if len(features.shape) == 1:
                features = np.expand_dims(features, 0)
            feed_dict = {self.generator_features: features}
            reward = sess.run(self.reward_op, feed_dict)
            return reward

        def train(generator_features: np.array, expert_features: np.array):
            feed_dict = {
                self.generator_features: generator_features,
                self.expert_features: expert_features
            }
            return sess.run([entropy, _train, *self.losses], feed_dict)

        def compute_accuracies(self, generator_features: np.array, expert_features: np.array):
            feed_dict = {
                self.generator_features: generator_features,
                self.expert_features: expert_features
            }
            return sess.run([generator_acc, expert_acc], feed_dict)

        self.classify = classify
        self.train = train
        self.compute_accuracies = compute_accuracies
        tf.global_variables_initializer().run(session=sess)

    def get_trainable_variables(self):
        with tf.variable_scope(self.scope):
            return tf.trainable_variables()

    @staticmethod
    def logsigmoid(a):
        """Equivalent to tf.log(tf.sigmoid(a))"""
        return -tf.nn.softplus(-a)

    @staticmethod
    def logit_bernoulli_entropy(logits):
        """Reference: https://github.com/openai/imitation/blob/99fbccf3e060b6e6c739bdf209758620fcdefd3c/policyopt/thutil
            .py#L48-L51"""
        ent = (1. - tf.nn.sigmoid(logits)) * logits - Discriminator.logsigmoid(logits)
        return ent


class Generator(object):
    """
    Generator policy training via PPO.
    """

    def __init__(self, *, policy_fn: PolicyFunction, ob_space, ac_space, nbatch_act, nbatch_train,
                 nsteps, ent_coef, vf_coef, max_grad_norm):
        sess = tf.get_default_session()

        act_model = policy_fn(sess, ob_space, ac_space, nbatch_act, 1, reuse=False)
        train_model = policy_fn(sess, ob_space, ac_space, nbatch_train, nsteps, reuse=True)

        A = train_model.pdtype.sample_placeholder([None], name="A")
        ADV = tf.placeholder(tf.float32, [None], name="ADV")
        R = tf.placeholder(tf.float32, [None], name="R")
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None], name="OLDNEGLOGPAC")
        OLDVPRED = tf.placeholder(tf.float32, [None], name="OLDVPRED")
        LR = tf.placeholder(tf.float32, [], name="LR")
        CLIPRANGE = tf.placeholder(tf.float32, [], name="CLIPRANGE")

        neglogpac = train_model.pd.neglogp(A)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        with tf.variable_scope('model'):
            params = tf.trainable_variables()
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        _train = trainer.apply_gradients(grads)

        def train(lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
            # PPO runs on the GPU
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            td_map = {train_model.obs: obs, A: actions, ADV: advs, R: returns, LR: lr,
                      CLIPRANGE: cliprange, OLDNEGLOGPAC: neglogpacs, OLDVPRED: values}
            if states is not None:
                # in case we have a recurrent policy
                td_map[train_model.states] = states
                td_map[train_model.masks] = masks
            return sess.run(
                [pg_loss, vf_loss, entropy, approxkl, clipfrac, _train],
                td_map
            )[:-1]

        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

        def save(save_path):
            ps = sess.run(params)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)
            # If you want to load weights, also save/load observation scaling inside VecNormalize

        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess)


class GAIL:
    """
    Generative Adversarial Imitation Learning.
    """

    @log_parameters
    def __init__(self,
                 env: gym.Env,
                 policy_fn: PolicyFunction,
                 expert_rollouts: np.array,
                 descriptor: Callable[
                     [EnvType, Policy, int, np.array, np.array, float], np.array] = observations_and_actions,
                 discriminator_fn: Discriminator.__class__ = Discriminator,
                 generator_fn: Generator.__class__ = Generator,
                 episode_length: int = 2048,
                 lr: float = 3e-4,
                 ent_coef: float = 0.,
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 gamma: float = 0.99,
                 lam: float = 0.95,
                 cliprange: float = 0.2,
                 randomize_expert_rollouts: bool = False):
        self.descriptor = descriptor
        self.policy_fn = policy_fn
        self.raw_env = env
        # The point of VecEnv is to support multiple envs in which the agent is trained.
        # TODO leverage VecEnv and require a list of environments as input?
        self.env = DummyVecEnv([lambda: bench.Monitor(env, logger.get_dir(), allow_early_resets=True)])
        self.env = VecNormalize(self.env)
        self.expert_rollouts = np.array(expert_rollouts)
        self.num_experts = self.expert_rollouts.shape[0]
        self.randomize_expert_rollouts = randomize_expert_rollouts
        self.expert_batch_pointer = 0

        self.nenvs = self.env.num_envs
        self.ob_space = self.env.observation_space
        self.ac_space = self.env.action_space

        sess = tf.get_default_session()

        # compute dummy features to obtain feature dimensions
        dummy_ob = env.reset()
        dummy_pi = policy_fn(sess, self.ob_space, self.ac_space, 1, 1, False, 'dummy_')
        tf.global_variables_initializer().run(session=sess)
        dummy_acs, _, _, _ = dummy_pi.step([dummy_ob])
        dummy_ft = descriptor(self.env, dummy_pi, 0, dummy_ob, dummy_acs[0], 0)
        logger.logkv("observation space", self.ob_space.shape[0])
        logger.logkv("action space", self.ac_space.shape[0])
        logger.logkv("feature space", dummy_ft.shape[0])
        logger.dumpkvs()
        self.ft_shape = dummy_ft.shape
        assert self.expert_rollouts.shape[1] == max(self.ft_shape)
        if self.expert_rollouts.shape[0] < episode_length:
            logger.warn("Length of expert rollouts is shorter than desired episode length.")

        self.discriminator_fn = discriminator_fn
        self.generator_fn = generator_fn

        self.discriminator_instance_noise_std = 0.2

        self.episode_length = episode_length
        self.lr = lr
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.lam = lam
        self.cliprange = cliprange

        if isinstance(self.lr, float):
            self.lr = utils.constfn(self.lr)
        else:
            assert callable(self.lr)
        if isinstance(self.cliprange, float):
            self.cliprange = utils.constfn(self.cliprange)
        else:
            assert callable(self.cliprange)

        self.generator = None
        self.discriminator = None

    @staticmethod
    def _sf01(arr):
        """
        Swap and then flatten axes 0 and 1.
        """
        s = arr.shape
        return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

    def _run_generator(self):
        """
        Computes generator's rollout rewarded by the discriminator.
        :return: Rollout stats of size episode_length for obs, features, rewards, etc.
        """
        mb_obs, mb_features, mb_rewards, mb_env_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], [], [], [], []
        mb_states = self.generator.initial_state
        dones = [False for _ in range(self.nenvs)]
        obs = np.zeros((self.nenvs,) + self.ob_space.shape, dtype=self.generator.train_model.obs.dtype.name)
        obs[:] = self.env.reset()
        epinfos = []
        for time_step in range(self.episode_length):
            actions, values, mb_states, neglogpacs = self.generator.step(obs, mb_states, dones)
            mb_obs.append(obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(dones)
            obs[:], env_rewards, dones, infos = self.env.step(actions)
            mb_env_rewards.append(env_rewards)
            # compute discriminator's reward
            features = self.descriptor(self.env, self.generator.act_model, time_step, obs, actions, env_rewards)
            mb_features.append(features)
            rewards = self.discriminator.classify(features)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo:
                    epinfos.append(maybeepinfo)
            mb_rewards.append(rewards[0])
        # batch of steps to batch of rollouts
        mb_obs = np.array(mb_obs, dtype=np.float32)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_env_rewards = np.asarray(mb_env_rewards, dtype=np.float32)
        mb_features = np.asarray(mb_features, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.generator.value(obs, mb_states, dones)
        # discount/bootstrap off value fn
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.episode_length)):
            if t == self.episode_length - 1:
                nextnonterminal = 1.0 - dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t + 1]
                nextvalues = mb_values[t + 1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return (*map(GAIL._sf01, (mb_obs, mb_features, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
                mb_states, epinfos)

    def _next_expert_batch(self, length: int) -> np.array:
        """
        Considers expert rollouts as circular buffer and returns (randomized) batch of desired length.
        """
        # from current pointer to end of experts
        batch = self.expert_rollouts[
                self.expert_batch_pointer:min(self.num_experts, self.expert_batch_pointer + length)]
        start_size = batch.shape[0]
        self.expert_batch_pointer = (self.expert_batch_pointer + start_size) % self.num_experts
        if start_size == length:
            return batch
        assert self.expert_batch_pointer == 0
        if self.randomize_expert_rollouts:
            np.random.shuffle(self.expert_rollouts)
        # repeat expert sequence
        for _ in range((length - start_size) // self.num_experts):
            batch = np.concatenate((batch, self.expert_rollouts), axis=0)
            if self.randomize_expert_rollouts:
                np.random.shuffle(self.expert_rollouts)
        # from start of experts to desired batch end
        self.expert_batch_pointer = (length - start_size) % self.num_experts
        batch = np.concatenate((batch, self.expert_rollouts[:self.expert_batch_pointer]), axis=0)
        assert batch.shape[0] == length
        return batch

    @log_parameters
    def train(self,
              total_timesteps: int,
              nminibatches: int = 4,
              noptepochs: int = 4,
              log_interval: int = 10,
              save_interval: int = 0,
              discriminator_training_rounds: int = 1,
              discriminator_instance_noise_acc_threshold: float = 0.95,
              callback: Callable[[object, int, EnvType, Policy, dict, dict], None] = lambda **args: None) -> None:
        """
        Trains generator and discriminator.
        :param discriminator_instance_noise_acc_threshold:
        :param discriminator_training_rounds:
        :param callback:
        :param total_timesteps: Number of training iterations.
        :param nminibatches:
        :param noptepochs:
        :param log_interval:
        :param save_interval:
        """

        nbatch = self.nenvs * self.episode_length
        nbatch_train = nbatch // nminibatches

        make_generator = lambda: self.generator_fn(
            policy_fn=self.policy_fn,
            ob_space=self.ob_space,
            ac_space=self.ac_space,
            nbatch_act=self.nenvs,
            nbatch_train=nbatch_train,
            nsteps=self.episode_length,
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm)
        if save_interval and logger.get_dir():
            import cloudpickle
            with open(osp.join(logger.get_dir(), 'make_generator.pkl'), 'wb') as fh:
                fh.write(cloudpickle.dumps(make_generator))

        self.generator = make_generator()
        self.discriminator = self.discriminator_fn(self.ft_shape)

        epinfobuf = deque(maxlen=100)
        tfirststart = time.time()

        nupdates = total_timesteps // nbatch
        for update in range(1, nupdates + 1):
            assert nbatch % nminibatches == 0
            nbatch_train = nbatch // nminibatches
            tstart = time.time()
            frac = 1.0 - (update - 1.0) / nupdates
            lrnow = self.lr(frac)
            cliprangenow = self.cliprange(frac)

            # compute rollout from generator
            # noinspection PyTupleAssignmentBalance
            obs, generator_features, returns, masks, actions, values, neglogpacs, states, epinfos = self._run_generator()

            #
            # train generator
            #
            epinfobuf.extend(epinfos)
            mblossvals = []
            if states is None:
                # nonrecurrent version
                inds = np.arange(nbatch)
                for _ in range(noptepochs):
                    np.random.shuffle(inds)
                    for start in range(0, nbatch, nbatch_train):
                        end = start + nbatch_train
                        mbinds = inds[start:end]
                        slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                        mblossvals.append(self.generator.train(lrnow, cliprangenow, *slices))
            else:
                # recurrent version
                assert self.nenvs % nminibatches == 0
                envsperbatch = self.nenvs // nminibatches
                envinds = np.arange(self.nenvs)
                flatinds = np.arange(self.nenvs * self.episode_length).reshape(self.nenvs, self.episode_length)
                envsperbatch = nbatch_train // self.episode_length
                for _ in range(noptepochs):
                    np.random.shuffle(envinds)
                    for start in range(0, self.nenvs, envsperbatch):
                        end = start + envsperbatch
                        mbenvinds = envinds[start:end]
                        mbflatinds = flatinds[mbenvinds].ravel()
                        slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                        mbstates = states[mbenvinds]
                        mblossvals.append(self.generator.train(lrnow, cliprangenow, *slices, mbstates))

            #
            # train discriminator
            #
            discriminator_losses = np.zeros((discriminator_training_rounds, 6))
            for dis_round in range(discriminator_training_rounds):
                expert_features = self._next_expert_batch(obs.shape[0])
                # update running mean/std for reward_giver
                # if hasattr(reward_giver, "obs_rms"):
                #     reward_giver.obs_rms.update(np.concatenate((ob_batch, ob_expert), 0))
                # *newlosses, g = reward_giver.lossandgrad(ob_batch, ac_batch, ob_expert, ac_expert)
                # d_adam.update(allmean(g), d_stepsize)
                # d_losses.append(newlosses)
                # logger.log(fmt_row(13, np.mean(d_losses, axis=0)))
                g_noise = np.random.normal(0.0, self.discriminator_instance_noise_std, expert_features.shape)
                e_noise = np.random.normal(0.0, self.discriminator_instance_noise_std, expert_features.shape)
                _, _, *dis_losses = self.discriminator.train(
                    generator_features + g_noise,
                    expert_features + e_noise)
                discriminator_losses[dis_round, :] = np.array(dis_losses)

            discriminator_losses = np.mean(discriminator_losses, axis=0)
            dis_accuracy = .5*(discriminator_losses[4]+discriminator_losses[5])
            if dis_accuracy > discriminator_instance_noise_acc_threshold:
                self.discriminator_instance_noise_std *= 2.
            elif dis_accuracy < discriminator_instance_noise_acc_threshold * 0.7:
                self.discriminator_instance_noise_std /= 2.

            lossvals = np.mean(mblossvals, axis=0)
            tnow = time.time()
            fps = int(nbatch / (tnow - tstart))
            if update % log_interval == 0 or update == 1:
                logger.logkv("serial_timesteps", update * self.episode_length)
                logger.logkv("nupdates", update)
                logger.logkv("total_timesteps", update * nbatch)
                logger.logkv("fps", fps)

                logger.logkv("dis_accuracy", dis_accuracy)
                logger.logkv("dis_loss", .5*(discriminator_losses[0]+discriminator_losses[1]))
                logger.logkv("dis_entropy", discriminator_losses[2])
                logger.logkv("dis_entloss", discriminator_losses[3])
                logger.logkv("dis_noise_std", self.discriminator_instance_noise_std)

                ev = explained_variance(values, returns)
                logger.logkv("explained_variance", float(ev))
                logger.logkv('epdisretmean', utils.safemean(returns))
                logger.logkv('epenvrewmean', utils.safemean([epinfo['r'] for epinfo in epinfobuf]))
                logger.logkv('eplenmean', utils.safemean([epinfo['l'] for epinfo in epinfobuf]))
                logger.logkv('time_elapsed', tnow - tfirststart)
                for (lossval, lossname) in zip(lossvals, self.generator.loss_names):
                    logger.logkv('gen_' + lossname, lossval)
                logger.dumpkvs()
            if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
                checkdir = osp.join(logger.get_dir(), 'checkpoints')
                os.makedirs(checkdir, exist_ok=True)
                savepath = osp.join(checkdir, '%.5i' % update)
                print('Saving generator to', savepath)
                self.generator.save(savepath)

            callback(self, update, self.raw_env, self.generator.act_model, locals(), globals())

        self.env.close()


@log_parameters
def make_policy(hidden_layers=(64, 64), hidden_activation=tf.tanh):
    class CustomMlpPolicy(Policy):
        def __init__(self, sess, ob_space, ac_space, nbatch, _nsteps, reuse=False, name_prefix=""):
            super().__init__()
            self.sess = sess
            ob_shape = (nbatch,) + ob_space.shape
            actdim = ac_space.shape[0]
            self.obs = tf.placeholder(tf.float32, ob_shape, name=name_prefix + 'obs')
            with tf.variable_scope(name_prefix + "model", reuse=reuse):
                hidden = self.obs
                for i, size in enumerate(hidden_layers):
                    hidden = hidden_activation(fc(self.obs, 'pi_fc%i' % i, nh=size, init_scale=np.sqrt(2)))
                pi = fc(hidden, 'pi', actdim, init_scale=0.01)
                hidden = self.obs
                for i, size in enumerate(hidden_layers):
                    hidden = hidden_activation(fc(self.obs, 'vf_fc%i' % i, nh=size, init_scale=np.sqrt(2)))
                vf = fc(hidden, 'vf', 1)[:, 0]
                logstd = tf.get_variable(name="logstd", shape=[1, actdim],
                                         initializer=tf.zeros_initializer())

            pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)

            self.pdtype = make_pdtype(ac_space)
            self.pd = self.pdtype.pdfromflat(pdparam)

            self.a0 = self.pd.sample()
            self.neglogp0 = self.pd.neglogp(self.a0)
            self.initial_state = None

            self.pi = pi
            self.vf = vf

        def step(self, ob, *_args, **_kwargs):
            a, v, neglogp = self.sess.run([self.a0, self.vf, self.neglogp0], {self.obs: ob})
            return a, v, self.initial_state, neglogp

        def value(self, ob, *_args, **_kwargs):
            return self.sess.run(self.vf, {self.obs: ob})

    return CustomMlpPolicy


def make_video_saver(folder_name: str, prefix: str, episode_length: int = 500, interval: int = 5,
                     video_width: int = 400, video_height: int = 400, plot_rewards: bool = True,
                     break_when_done: bool = False):
    def save_video(gail: GAIL, iteration: int, env: gym.Env, policy: Policy, _locals: dict, _globals: dict):
        if iteration % interval != 0:
            return
        print('Saving video at iteration %i...' % iteration)

        fps = env.metadata.get('video.frames_per_second', 50)
        modes = env.metadata.get('render.modes', [])

        images = []
        rewards = []
        returns = []
        max_reward = 1.  # if any reward > 1, we have to rescale
        max_return = 1.
        lower_part = video_height // 5

        obs = env.reset()
        for step in range(episode_length):
            action, _, _, _ = policy.step([obs])
            obs, rew, done, _ = env.step(action)
            if 'rgb_array' not in modes:
                env.render()
            else:
                # TODO add support for rendering multiple cameras (as in dm_control envs)
                images.append((env.render(mode='rgb_array'),))
            if plot_rewards:
                # environment rewards
                rewards.append(rew)
                max_reward = max(rew, max_reward)
                # discriminator returns
                features = gail.descriptor(gail.env, policy, iteration, [obs], action, rew)
                rets = gail.discriminator.classify(features)
                returns.append(rets[0])
                max_return = max(rets[0], max_return)
            if break_when_done and done:
                break

        orange = np.array([255, 163, 0])
        green = np.array([100, 200, 50])
        red = np.array([255, 0, 0])
        video = []
        width_factor = 1. / episode_length * video_width
        for i, imgs in enumerate(images):
            for img in imgs:
                img[-lower_part, :10] = orange
                img[-lower_part, -10:] = orange
                if episode_length < video_width:
                    p_r_x = 0
                    for j, (rew, ret) in enumerate(zip(rewards[:i], returns[:i])):
                        r_x = int(j * width_factor)
                        # render environment reward
                        if rew < 0:
                            img[-1:, p_r_x:r_x] = red
                            img[-1:, p_r_x:r_x] = red
                        else:
                            rew_y = int(rew / max_reward * lower_part)
                            img[-rew_y - 1:, p_r_x:r_x] = orange
                            img[-rew_y - 1:, p_r_x:r_x] = orange
                        # render discriminator reward
                        if ret < 0:
                            img[-1:, p_r_x:r_x] = red
                            img[-1:, p_r_x:r_x] = red
                        else:
                            ret_y = int(ret / max_return * lower_part)
                            img[-ret_y - 1:, p_r_x:r_x] = green
                            img[-ret_y - 1:, p_r_x:r_x] = green
                        p_r_x = r_x
                else:
                    for j, (rew, ret) in enumerate(zip(rewards[:i], returns[:i])):
                        r_x = int(j * width_factor)
                        # render environment reward
                        if rew < 0:
                            img[-1:, r_x] = red
                            img[-1:, r_x] = red
                        else:
                            rew_y = int(rew / max_reward * lower_part)
                            img[-rew_y - 1:, r_x] = orange
                            img[-rew_y - 1:, r_x] = orange
                        # render discriminator reward
                        if ret < 0:
                            img[-1:, r_x] = red
                            img[-1:, r_x] = red
                        else:
                            ret_y = int(ret / max_return * lower_part)
                            img[-ret_y - 1:, r_x] = green
                            img[-ret_y - 1:, r_x] = green
            video.append(np.hstack(imgs))
        imageio.mimsave(
            os.path.join(folder_name, "%s_iteration_%i.mp4" % (prefix, iteration)),
            video,
            fps=fps)

    return save_video


def main(num_timesteps: int, environment: str, expert_rollouts: str, **_):
    # load expert rollouts
    rollouts = pkl.load(open(expert_rollouts, "rb"))
    expert_rollouts = []
    for rollout in tqdm(rollouts, desc="Loading expert rollouts..."):
        # here we are using observation-action pairs as features for the discriminator
        for observation, action in zip(rollout["observation"], rollout["action"]):
            expert_rollouts.append(np.hstack((observation, action)))

    def run_gail(seed: int):
        env = utils.create_environment(environment)
        set_global_seeds(seed)

        policy_fn = make_policy(hidden_layers=(64, 64), hidden_activation=tf.tanh)
        gail = GAIL(env,
                    policy_fn,
                    expert_rollouts,
                    episode_length=2048,
                    lam=0.95,
                    gamma=0.99,
                    ent_coef=0.0,
                    lr=3e-4,
                    cliprange=0.2)
        video_saver = make_video_saver(
            folder_name=osp.join(logger.get_dir(), 'videos'),
            prefix='gail',
            interval=20
        )
        gail.train(nminibatches=32,
                   noptepochs=10,
                   log_interval=1,
                   total_timesteps=num_timesteps,
                   callback=video_saver)

    experiment = Experiment('GAIL')
    num_cpu = 4
    experiment.run(run_gail, num_cpu)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    parser.add_argument(
        '--environment',
        help='environment ID prefixed by framework, e.g. dm-cartpole-swingup, gym-CartPole-v0, rllab-cartpole',
        type=str,
        default='gym-Hopper-v2')
    parser.add_argument(
        '--expert-rollouts',
        help='path to a Pickle file containing expert trajectories',
        type=str,
        default='logs/rollouts.pkl')
    # gym-Hopper-v2
    # default='rllab-humanoid')

    parser.add_argument(
        '--logdir',
        help='folder where the logs will be stored',
        type=str,
        default='logs')

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
