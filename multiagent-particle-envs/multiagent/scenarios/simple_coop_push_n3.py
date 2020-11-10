import numpy as np
import random
from multiagent.core_vec import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, sort_obs=True):
        world = World()
        self.np_rnd = np.random.RandomState(0)
        self.sort_obs = sort_obs
        # set any world properties first
        world.dim_c = 2
        num_agents = 3
        world.num_adversaries = 0
        num_landmarks = 2
        world.collaborative = True
        self.world_radius = 1
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.06
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            if i < num_landmarks / 2:
                landmark.name = 'landmark %d' % i
                landmark.collide = True
                landmark.movable = True
                landmark.size = 0.1
                landmark.initial_mass = 2.0
            else:
                landmark.name = 'target %d' % (i - num_landmarks / 2)
                landmark.collide = False
                landmark.movable = False
                landmark.size = 0.05
                landmark.initial_mass = 4.0
        # make initial conditions
        self.color = {
                      'green': np.array([0.35, 0.85, 0.35]), 'blue': np.array([0.35, 0.35, 0.85]),'red': np.array([0.85, 0.35, 0.35]),
                      'light_blue': np.array([0.35, 0.85, 0.85]), 'yellow': np.array([0.85, 0.85, 0.35]), 'black': np.array([0.0, 0.0, 0.0])}
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.0, 0.0, 0.0])
        # random properties for landmarks
        color_keys = list(self.color.keys())
        for i, landmark in enumerate(world.landmarks):
            if i < len(world.landmarks) / 2:
                landmark.color = self.color[color_keys[i]] - 0.1
            else:
                landmark.color = self.color[color_keys[int(i / 2)]] + 0.1
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = self.np_rnd.uniform(-self.world_radius, +self.world_radius, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        num_landmark = int(len(world.landmarks) / 2)
        for i, landmark in enumerate(world.landmarks[:num_landmark]):
            landmark.state.p_pos = self.np_rnd.uniform(-(self.world_radius - 0.2) , +self.world_radius - 0.2, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        for i, target in enumerate(world.landmarks[num_landmark:]):
            target.state.p_pos = self.np_rnd.uniform(-(self.world_radius - 0.2) , +self.world_radius - 0.2, world.dim_p)
            while np.sqrt(np.sum(np.square(target.state.p_pos - world.landmarks[i].state.p_pos))) < 0.8:
                target.state.p_pos = self.np_rnd.uniform(-(self.world_radius - 0.2), +self.world_radius - 0.2,
                                                         world.dim_p)
            target.state.p_vel = np.zeros(world.dim_p)

    # def benchmark_data(self, agent, world):
    #     # rew = 0
    #     collisions = 0
    #     occupied_landmarks = 0
    #     min_dists = 0
    #     for l in world.landmarks:
    #         dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
    #         min_dists += min(dists)
    #         # rew -= min(dists)
    #         if min(dists) < 0.1:
    #             occupied_landmarks += 1
    #     if agent.collide:
    #         for a in world.agents:
    #             if a != agent and self.is_collision(a, agent):
    #                 # rew -= 1
    #                 collisions += 1
    #     # return [rew, collisions, min_dists, occupied_landmarks]
    #     return [collisions / 2, min_dists, occupied_landmarks]
    def benchmark_data(self, agent, world):
        collisions = 0.
        occupied_target = 0.
        avg_dist = 0.
        num_landmark = int(len(world.landmarks) / 2)
        for l, t in zip(world.landmarks[:num_landmark],
                        world.landmarks[num_landmark:]):
            dist = np.sqrt(np.sum(np.square(l.state.p_pos - t.state.p_pos)))
            avg_dist += dist / num_landmark
            if dist < 0.1:
                occupied_target += 1
            if agent.collide:
                if self.is_collision(agent, l):
                    collisions += 1
        return collisions, avg_dist, occupied_target

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # def reward(self, agent, world):
    #     # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
    #     rew = 0
    #     if agent == world.agents[0]:
    #         num_landmark = int(len(world.landmarks) / 2)
    #         for l, t in zip(world.landmarks[:num_landmark], world.landmarks[num_landmark:]):
    #             dist = np.sqrt(np.sum(np.square(l.state.p_pos - t.state.p_pos)))
    #             rew -= 2 * dist
    #             dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
    #             rew -= 0.1 * min(dists)
    #             for a in world.agents:
    #                 if self.is_collision(l, a):
    #                     rew += 0.1
    #     return rew
    def reward(self, agent, world):
        rew = 0.0
        num_landmark = int(len(world.landmarks) / 2)
        shape = False
        bound_only = False
        if not bound_only:
            if shape:
                dists = [1 / np.sqrt(np.sum(np.square(
                         agent.state.p_pos - l.state.p_pos)))
                         for l in world.landmarks[:num_landmark]]
                rew += max(dists)
                for l, t in zip(world.landmarks[:num_landmark],
                                world.landmarks[num_landmark:]):
                    if agent.collide and self.is_collision(agent, l):
                        rew += 20
                    rew -= 10 * np.sqrt(np.sum(np.square(
                            t.state.p_pos - l.state.p_pos)))
                    if self.is_collision(l, t):
                        rew += 20
            else:
                for l, t in zip(world.landmarks[:num_landmark],
                                world.landmarks[num_landmark:]):
                    dist = np.sqrt(np.sum(np.square(
                            l.state.p_pos - t.state.p_pos)))
                    rew -= 10 * dist
                    if self.is_collision(l, t):
                        rew += 200
                    dists = [1 / np.sqrt(np.sum(np.square(
                             a.state.p_pos - l.state.p_pos)))
                             for a in world.agents]
                    rew += max(dists)
                    for a in world.agents:
                        if self.is_collision(l, a):
                            rew += 0.1

        # agents are penalized for exiting the screen
        def bound(x):
            if x < 0.9:
                return 0.0
            if x < 1.0:
                return (x - 0.9) * 10
            # return min(np.exp(2 * x - 2), 10)  # 1 + (x - 1) * (x - 1)
            # not sure why 2 * (x - 1) is necessary to replacing it with 1 *
            return min(np.exp(x - 1), 10)
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            agent_bound_penalty = bound(x)
            # rew -= agent_bound_penalty

        if bound_only:
            dists = [np.sqrt(np.sum(np.square(
                    agent.state.p_pos - l.state.p_pos)))
                    for l in world.landmarks[:num_landmark]]
            rew += 1.0 - min(dists)
        return rew

    def observation(self, agent, world):
        """
        :param agent: an agent
        :param world: the current world
        :return: obs: [18] np array,
        [0-1] self_agent velocity
        [2-3] self_agent location
        [4-9] landmarks location
        [10-11] agent_i's relative location
        [12-13] agent_j's relative location
        Note that i < j
        """
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        obs = np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos)
        return obs

    def seed(self, seed=None):
        self.np_rnd.seed(seed)
