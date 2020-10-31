import csv
import torch 
import numpy as np
from cv2 import VideoWriter, VideoWriter_fourcc

def create_video(frames, video_file):
    width, height, _ = frames[0].shape
    fps = 60
    # fourcc = VideoWriter_fourcc(*'MP42')
    fourcc = VideoWriter_fourcc(*'MP4V')
    video = VideoWriter(video_file, fourcc, float(fps), (width, height))

    for i in range(len(frames)):
        video.write(frames[i])
    video.release()


def adjust_learning_rate(optimizer, steps, max_steps, start_decrease_step, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if steps > start_decrease_step:
        lr = init_lr * (1 - ((steps - start_decrease_step) / (max_steps - start_decrease_step)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def dict2csv(output_dict, f_name):
    with open(f_name, mode='w') as f:
        writer = csv.writer(f, delimiter=",")
        for k, v in output_dict.items():
            v = [k] + v
            writer.writerow(v)


def n_actions(action_spaces):
    """
    :param action_space: list
    :return: n_action: list
    """
    n_actions = []
    from gym import spaces
    from multiagent.environment import MultiDiscrete
    for action_space in action_spaces:
        if isinstance(action_space, spaces.discrete.Discrete):
            n_actions.append(action_space.n)
        elif isinstance(action_space, MultiDiscrete):
            total_n_action = 0
            one_agent_n_action = 0
            for h, l in zip(action_space.high, action_space.low):
                total_n_action += int(h - l + 1)
                one_agent_n_action += int(h - l + 1)
            n_actions.append(one_agent_n_action)
        else:
            raise NotImplementedError
    return n_actions


def grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data,
                            seed_callback=scenario.seed, cam_range=scenario.world_radius)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                            seed_callback=scenario.seed, cam_range=scenario.world_radius)
    return env





def make_env_vec(scenario_name, arglist, benchmark=False):
    from multiagent.environment_vec import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data,
                            seed_callback=scenario.seed, cam_range=scenario.world_radius)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                            seed_callback=scenario.seed, cam_range=scenario.world_radius)
    return env

def copy_actor_policy(s_agent, t_agent):
    if hasattr(s_agent, 'actors'):
        for i in range(s_agent.n_group):
            state_dict = s_agent.actors[i].state_dict()
            for k, v in state_dict.items():
                state_dict[k] = v.cpu()
            t_agent.actors[i].load_state_dict(state_dict)
        t_agent.actors_params, t_agent.critic_params = None, None
    else:
        state_dict = s_agent.actor.state_dict()
        for k, v in state_dict.items():
            state_dict[k] = v.cpu()
        t_agent.actor.load_state_dict(state_dict)
        t_agent.actor_params, t_agent.critic_params = None, None

def calculate_alignment_reward(batch, memory, indice, extra_rew, num_agents, num_adversaries):
    import time 
    t1 = time.time()
    batch_size = len(batch.state)
    buf_len = len(memory)
    action_n = int(memory[0].action.size()[1] / num_agents)
    # dones, acts = [], []
    # for i in range(buf_len):
    #     dones.append(memory[i].mask)
    #     act_probs = memory[i].action
    #     act = torch.zeros((1, num_agents))
    #     for j in range(num_agents):
    #         act[0, j] = torch.argmax(act_probs[0, j:j+action_n])
    #     acts.append(act)
    # done = torch.cat(dones, 0)
    # act = torch.cat(acts, 0)
    # reward = torch.cat([batch.reward[i] for i in range(batch_size)],0)
    # true = torch.FloatTensor([1.0]) # false is 1.0 and true 0.0
    # trues = true.repeat(1,batch_size)
    
    done = np.asarray([memory[i].mask.detach().numpy()[0] for i in range(buf_len)])
    act = np.asarray([[np.argmax(memory[j].action.detach().numpy()[0][i:i+action_n]) for i in range(num_agents)] for j in range(buf_len)])
    reward = np.asarray([batch.reward[i].detach().numpy()[0] for i in range(batch_size)])
    trues = np.asarray([True] * batch_size)

    now = indice % buf_len
    last = (indice - 1) % buf_len
    t2 = time.time()
    for k in range(num_adversaries + 1, num_agents):
        # get a mask to encode where the last action is not the end of the episode
        done_mask = done[last,k] == trues
        act_mask = act[now, k] == act[last, num_adversaries]
        combined_mask = done_mask & act_mask
        # combined_mask = torch.reshape(combined_mask, (batch_size,))
        reward[:, k][combined_mask] += extra_rew 
    t3 = time.time()
    new_reward = tuple(torch.reshape(torch.Tensor(reward[i, :]), (1, num_agents)) for i in range(batch_size))
    # new_reward = tuple(torch.reshape(reward[i, :], (1, num_agents)) for i in range(batch_size))
    t4 = time.time()
    print(t2 - t1, t3 - t2, t4 - t3, t4 - t1)
    return batch._replace(reward=new_reward)

def process_fn(batch, memory, indice, **kwargs):
    params = {}
    for key, value in kwargs.items():
        params[key] = value
    extra_rew = params['extra_rew'] if 'extra_rew' in params else 0.0
    if extra_rew == 0.0: return batch
    num_adversaries = params['num_adversaries'] if 'num_adversaries' in params else 0
    num_agents = params['num_agents'] if 'num_adversaries' in params else 0
    batch = calculate_alignment_reward(
            batch, memory, indice, extra_rew, num_agents, num_adversaries)
    return batch

