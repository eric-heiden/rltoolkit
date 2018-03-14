from tqdm import tqdm

import utils, sys, json, re, glob, operator, imageio
import tensorflow as tf
import os.path as osp
import pickle as pkl
import numpy as np

sys.path.insert(0, osp.join(osp.dirname(__file__), 'baselines'))
sys.path.insert(0, osp.join(osp.dirname(__file__), 'rllab'))
sys.path.insert(0, osp.join(osp.dirname(__file__), 'softqlearning'))
utils.load_dm_control()

from baselines.common.misc_util import boolean_flag


def main(logdir, checkpoint, human_render, num_rollouts, max_episode_length, save_videos, save_rollouts, save_separate_rollouts):
    if not osp.exists(osp.join(logdir, 'run.json')):
        raise FileNotFoundError("Could not find run.json.")

    configuration = json.load(open(osp.join(logdir, 'run.json'), 'r'))
    if configuration["settings"]["method"] not in ["trpo", "ppo"]:
        raise NotImplementedError("Playback for %s has not been implemented yet." % configuration["method"])

    env = utils.create_environment(configuration["settings"]["environment"])

    # build policy network
    # TODO this needs to be more general
    from baselines.ppo1 import mlp_policy
    tf.Session().__enter__()
    pi = mlp_policy.MlpPolicy(
        name="pi",
        ob_space=env.observation_space,
        ac_space=env.action_space,
        hid_size=configuration["settings"].get('pi_hid_size', 150),
        num_hid_layers=configuration["settings"].get('pi_num_hid_layers', 3))

    # find latest policy checkpoint
    saver = tf.train.Saver()
    if checkpoint is None:
        files = glob.glob(osp.join(logdir, 'checkpoints') + '/*.index')
        files = [(int(re.findall(".*?_(\d+)\.", f)[0]), f) for f in files]
        files = sorted(files, key=operator.itemgetter(0))
        checkpoint = files[-1][1]
    elif not osp.isabs(checkpoint):
        if not osp.exists(osp.join(logdir, 'checkpoints')):
            raise FileNotFoundError("Could not find checkpoints folder")
        else:
            checkpoint = osp.join(logdir, 'checkpoints', checkpoint)
    if checkpoint.endswith(".index"):
        checkpoint = checkpoint[:-len(".index")]
    print("Loading checkpoint %s." % checkpoint)
    saver.restore(tf.get_default_session(), checkpoint)

    # generate rollouts
    rollouts = []
    for i_rollout in tqdm(range(num_rollouts), "Computing rollouts"):
        observation = env.reset()
        rollout = {
            "observation": [],
            "reward": [],
            "action": []
        }
        video = []
        for i_episode in range(max_episode_length):
            action, _ = pi.act(stochastic=False, ob=observation)
            observation, reward, done, _ = env.step(action)
            if human_render:
                env.render(mode='human')
            if save_videos is not None:
                video.append(env.render(mode='rgb_array'))
            if save_rollouts is not None:
                rollout["observation"].append(observation)
                rollout["reward"].append(reward)
                rollout["action"].append(action)
            if done:
                break

        if save_videos is not None:
            imageio.mimsave(
                osp.join(save_videos, 'rollout_%i.mp4' % i_rollout),
                video,
                fps=env.metadata.get('video.frames_per_second', 50))
        if save_rollouts is not None and save_separate_rollouts:
            pkl.dump(rollout, open(osp.join(save_rollouts, 'rollout_%i.pkl' % i_rollout), "wb"))
        else:
            rollouts.append(rollout)

    if save_rollouts is not None and not save_separate_rollouts:
        pkl.dump(rollouts, open(osp.join(save_rollouts, 'rollouts.pkl'), "wb"))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--logdir', type=str, default='logs/2018-03-11-00-37-42-gym-Hopper-v2-ppo')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--save-rollouts', type=str, default=None)
    boolean_flag(parser, 'save-separate-rollouts', default=False, help='Save all rollouts in one pickle file')
    parser.add_argument('--save-videos', type=str, default=None)
    boolean_flag(parser, 'human-render', default=True)
    parser.add_argument('--num-rollouts', type=int, default=10)
    parser.add_argument('--max-episode-length', type=int, default=1000)
    args = parser.parse_args()
    main(**vars(args))
