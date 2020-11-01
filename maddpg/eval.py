from utils import make_env, dict2csv
import numpy as np
import contextlib
import torch
from ckpt_plot.plot_curve import plot_result
import os
import time


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def eval_model_q(test_q, done_training, args):
    if 'simple_tag' in args.scenario:
        plot = {'good_rewards': [], 'adversary_rewards': [], 'rewards': [], 'collisions': [], 'dists': [], 'steps': [], 'q_loss': [], 'gcn_q_loss': [],
            'p_loss': [], 'final': [], 'abs': []}
    elif 'simple_coop_push' in args.scenario:
        plot = {'good_rewards': [], 'adversary_rewards': [], 'rewards': [], 'collisions': [], 'avg_dists': [], 'occupied_targets': [], 'steps': [], 'q_loss': [], 'gcn_q_loss': [],
            'p_loss': [], 'final': [], 'abs': []}
    else:
        plot = {'good_rewards': [], 'adversary_rewards': [], 'rewards': [], 'steps': [], 'q_loss': [], 'gcn_q_loss': [],
            'p_loss': [], 'final': [], 'abs': []}
    # if args.render:
    #     frame_data = {'frames': []}
    best_eval_reward = -100000000
    while True:
        if not test_q.empty():
            print('=================== start eval ===================')
            eval_env = make_env(args.scenario, args, benchmark=True)
            eval_env.seed(args.seed + 10)
            eval_rewards = []
            good_eval_rewards = []
            if 'simple_coop_push' in args.scenario:
                eval_occupied_targets = []
            eval_collisions = []
            eval_dists = []
            agent, tr_log = test_q.get()
            num_adversaries = eval_env.world.num_adversaries
            with temp_seed(args.seed):
                for n_eval in range(args.num_eval_runs):
                    obs_n = eval_env.reset()
                    episode_reward = 0
                    episode_step = 0
                    n_agents = eval_env.n
                    agents_rew = [[] for _ in range(n_agents)]
                    # frames = []
                    if 'simple_tag' in args.scenario:
                        episode_benchmark = [0 for _ in range(2)]
                    elif 'simple_coop_push' in args.scenario:
                        episode_benchmark = [0 for _ in range(3)]
                    while True:
                        action_n = agent.select_action(torch.Tensor(obs_n), action_noise=True,
                                                       param_noise=False).squeeze().cpu().numpy()
                        next_obs_n, reward_n, done_n, info_n = eval_env.step(action_n)
                        benchmark_n = np.asarray(info_n['n'])
                        episode_step += 1
                        if "simple_tag" in args.scenario:
                            # collisions for adversaries only
                            episode_benchmark[0] += sum(benchmark_n[:num_adversaries, 0])
                            # min distance for good agents only 
                            episode_benchmark[1] += sum(benchmark_n[num_adversaries:, 1])
                        elif 'simple_coop_push' in args.scenario:
                            for i in range(len(episode_benchmark)):
                                episode_benchmark[i] += sum(benchmark_n[:, i])
                        
                        # if args.render and n_eval % args.render_freq == 0:
                        #     if args.render_mode:
                        #         frame = eval_env.render(mode=args.render_mode)[0]
                        #         frames.append(frame)
                        #     else:
                        #         eval_env.render()
                        #     if args.render > 0:
                        #         time.sleep(args.render)
    
                        terminal = (episode_step >= args.num_steps)
                        
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
                            if 'simple_coop_push' in args.scenario:
                                eval_occupied_targets.append(episode_benchmark[2])
                            if n_eval % 100 == 0:
                                print('test reward', episode_reward)
                            break
                if np.mean(eval_rewards) > best_eval_reward:
                    best_eval_reward = np.mean(eval_rewards)
                    torch.save({'agents': agent}, os.path.join(tr_log['exp_save_dir'], 'agents_best.ckpt'))

                plot['rewards'].append(np.mean(eval_rewards))
                if 'simple_tag' in args.scenario:
                    plot['collisions'].append(np.mean(eval_collisions))
                    plot['dists'].append(np.mean(eval_dists))
                elif 'simple_coop_push' in args.scenario:
                    plot['collisions'].append(np.mean(eval_collisions))
                    plot['avg_dists'].append(np.mean(eval_dists))
                    plot['occupied_targets'].append(np.mean(eval_occupied_targets))
                plot['steps'].append(tr_log['total_numsteps'])
                plot['q_loss'].append(tr_log['value_loss'])
                plot['p_loss'].append(tr_log['policy_loss'])
                # if args.render and args.render_mode == 'rgb_array':
                #     frame_data['frames'] = frames
                print("========================================================")
                print(
                    "Episode: {}, total numsteps: {}, {} eval runs, total time: {} s".
                        format(tr_log['i_episode'], tr_log['total_numsteps'], args.num_eval_runs,
                               time.time() - tr_log['start_time']))
                print("GOOD reward: avg {} std {}, average reward: {}, best reward {}, \
                    average collision {}, average dist {}".format(np.mean(eval_rewards),
                    np.std(eval_rewards),np.mean(plot['rewards'][-10:]),best_eval_reward,
                    np.mean(eval_collisions),np.mean(eval_dists)))
                plot['final'].append(np.mean(plot['rewards'][-10:]))
                plot['abs'].append(best_eval_reward)
                dict2csv(plot, os.path.join(tr_log['exp_save_dir'], 'train_curve.csv'))
                # dict2csv(frame_data, os.path.join(tr_log['exp_save_dir'], 'frame_data.csv'))
                
                eval_env.close()
        if done_training.value and test_q.empty():
            torch.save({'agents': agent}, os.path.join(tr_log['exp_save_dir'], 'agents.ckpt'))
            break