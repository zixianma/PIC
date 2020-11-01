
# import argparse
# from ckpt_plot.plot_curve import read_csv
# from utils import create_video


# parser = argparse.ArgumentParser(description='Eval agent visualization')
# parser.add_argument('--video_file', type=str, default='videos/video.mp4',
#                     help='name of the environment to run')
# parser.add_argument('--frame_data_file', type=str, default='frame_data.csv',
#                     help='the path to the file storing frame data')
# args = parser.parse_args()

# data = read_csv(args.frame_data_file)
# num_render = len(data['frames'])
# for i in range(num_render):
#     create_video(data['frames'], args.video_file.replace(
#                         '.mp4', '_%d.mp4' % i))

# scripted_agent_ckpt = os.path.join(obs_path, 'scripted_agent_ckpt/simple_tag_v5_al0a10_4/agents.ckpt')
# self.scripted_agents = torch.load(scripted_agent_ckpt)['agents']
import sys
sys.path.append('/Users/zixianma/Desktop/Sophomore/Summer/CURIS/PIC/multiagent-particle-envs')
from utils import make_env, dict2csv, Dict2Obj, create_video
import numpy as np
import contextlib
import torch
from ckpt_plot.plot_curve import plot_result
import os
import time
import json
import argparse

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def get_args():
    parser = argparse.ArgumentParser()

    # State arguments.
    parser.add_argument('--video_file', type=str, default='videos/video.mp4')
    # parser.add_argument('--benchmark', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.03)
    parser.add_argument('--num_render', type=int, default=5)
    parser.add_argument('--render_mode', type=str, default='rgb_array')

    # parser.add_argument(
    #     '--device', type=str,
    #     default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_known_args()[0]
    return args

def visualize(args=get_args()):
    scripted_agent_ckpt = os.path.join(args.logdir, 'agents_best.ckpt')
    agent = torch.load(scripted_agent_ckpt)['agents']
    params = Dict2Obj(json.load(
            open(os.path.join(args.logdir, "args.json"), "r")))
    
    best_eval_reward = -100000000
    print('=================== start visualizing ===================')
    eval_env = make_env(params.scenario, params, benchmark=True)
    eval_env.seed(params.seed + 10)
    eval_rewards = []
    good_eval_rewards = []
    if 'simple_coop_push' in params.scenario:
        eval_occupied_targets = []
    eval_collisions = []
    eval_dists = []

    num_adversaries = eval_env.world.num_adversaries
    with temp_seed(params.seed):
        for n_render in range(args.num_render):
            if args.render:
                frame_data = {'frames': []}
            for n_eval in range(1):
                obs_n = eval_env.reset()
                episode_reward = 0
                episode_step = 0
                n_agents = eval_env.n
                agents_rew = [[] for _ in range(n_agents)]
                frames = []
                if 'simple_tag' in params.scenario:
                    episode_benchmark = [0 for _ in range(2)]
                elif 'simple_coop_push' in params.scenario:
                    episode_benchmark = [0 for _ in range(3)]
                while True:
                    action_n = agent.select_action(torch.Tensor(obs_n), action_noise=True,
                                                    param_noise=False).squeeze().cpu().numpy()
                    next_obs_n, reward_n, done_n, info_n = eval_env.step(action_n)
                    benchmark_n = np.asarray(info_n['n'])
                    episode_step += 1
                    if "simple_tag" in params.scenario:
                        # collisions for adversaries only
                        episode_benchmark[0] += sum(benchmark_n[:num_adversaries, 0])
                        # min distance for good agents only 
                        episode_benchmark[1] += sum(benchmark_n[num_adversaries:, 1])
                    elif 'simple_coop_push' in params.scenario:
                        for i in range(len(episode_benchmark)):
                            episode_benchmark[i] += sum(benchmark_n[:, i])
                    
                    if args.render:
                        if args.render_mode:
                            frame = eval_env.render(mode=args.render_mode)[0]
                            frames.append(frame)
                        else:
                            eval_env.render()
                        if args.render > 0:
                            time.sleep(args.render)

                    terminal = (episode_step >= params.num_steps)
                    
                    episode_reward += np.sum(reward_n)
                    for i, r in enumerate(reward_n):
                        agents_rew[i].append(r)
                    obs_n = next_obs_n
                    if done_n[0] or terminal:
                        eval_rewards.append(episode_reward)
                        agents_rew = [np.sum(rew) for rew in agents_rew]
                        good_reward = np.sum(agents_rew)
                        good_eval_rewards.append(good_reward)
                        eval_collisions.append(episode_benchmark[0])
                        eval_dists.append(episode_benchmark[1])
                        if 'simple_coop_push' in params.scenario:
                            eval_occupied_targets.append(episode_benchmark[2])
                        if n_eval % 100 == 0:
                            print('test reward', episode_reward)
                        break
                if np.mean(eval_rewards) > best_eval_reward:
                    best_eval_reward = np.mean(eval_rewards)
                    
                    print("========================================================")
                    print("GOOD reward: avg {} std {}, best reward {}, average collision {}, average dist {}" \
                        .format(np.mean(eval_rewards),
                        np.std(eval_rewards),best_eval_reward,
                        np.mean(eval_collisions),np.mean(eval_dists)))
                    
                    eval_env.close()

            if args.render and args.render_mode == 'rgb_array':
                    frame_data['frames'] = frames
            video_path = os.path.join(args.logdir, params.exp_name + ".mp4")
            create_video(frame_data['frames'], video_path.replace(
                        '.mp4', '_%d.mp4' % n_render))

if __name__ == "__main__":
    visualize()