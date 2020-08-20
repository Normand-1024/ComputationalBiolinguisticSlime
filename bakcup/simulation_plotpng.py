import random
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt

import util

class Agent:
    def __init__(self, pos = [0, 0, 0], direction = [0, 0, 0]):
        self.pos = np.array(pos)
        self.dir = np.array(direction)

    # Return sphere coordinates in r, theta, phi
    def get_sphere_coord(self):
        direc = self.dir

        r = np.sqrt(direc[0] ** 2 + direc[1] ** 2 + direc[2] ** 2)
        if (direc[0] >= 0):
            theta = np.arctan(direc[1] / direc[0]) % ( 2 * np.pi)
        else:
            theta = np.arctan(direc[1] / direc[0]) + np.pi
        phi = np.arccos(direc[2] / r)

        return [r, theta, phi]

    # axis-angle rotation
    def rotate(self, axis, angle):
        self.dir = util.axis_rotate(self.dir, axis, angle)

    def forward(self, dist):
        self.pos = self.pos + dist * self.dir

    def display(self):
        print("pos: {}, dir: {}".format(self.pos, self.dir))

class SlimeSimulation:
    def __init__(self, depo):
        self.depo = depo
        self.agents = []
        self.params = depo.params

    # Spawn agents, uniformly distributed in the box with random direction
    def initialize_agents(self, num_agents=None, seed=None, spawn_at=None):
        if (seed):
            np.random.seed(seed)
        if (not num_agents):
            num_agents = self.params["num_agent"]
        self.agents = []

        for i in range(num_agents):
            if (spawn_at is not None):
                pos = spawn_at
            else:
                pos = self.depo.grid_size / 2
                pos[1] = pos[0]
                
                """
                np.array([
                    np.random.uniform(0, self.depo.grid_size[0]),
                    np.random.uniform(0, self.depo.grid_size[1]),
                    np.random.uniform(0, self.depo.grid_size[2])
                ])
                """
            direc = np.random.normal(size=3)

            # Normalize direction
            norm = np.linalg.norm(direc)
            direc = direc / norm

            self.agents.append(Agent(pos=pos, direction=direc))

    def step(self):
        for agent in self.agents:
            # sense phase
            dir_s0 = agent.dir
            rotate_axis = util.rand_perpendicular(dir_s0)
            dir_s1 = util.axis_rotate(dir_s0,
                                    rotate_axis,
                                    self.params["sense_angle"])
            dist_s = random.uniform(0, self.params["sense_distance"])
            pos0 = agent.pos + dist_s * dir_s0
            pos1 = agent.pos + dist_s * dir_s1
            #print(self.depo.get_deposit(pos0))
            p0 = self.depo.get_deposit(pos0 % self.depo.grid_size) ** self.params["sharpness"]
            p1 = self.depo.get_deposit(pos1 % self.depo.grid_size) ** self.params["sharpness"]

            # sample phase
            #if (random.random() > 0.5):
            if (random.random() > p0 / (p0 + p1)):
                agent.rotate(rotate_axis,
                            self.params["move_angle"])
            dist_m = random.uniform(0, self.params["move_distance"])

            # update phase
            agent.forward(dist_m)

            # TODO: for rendering agent distribution only, to be deleted
            if (any(agent.pos > self.depo.grid_size) or
                any(agent.pos < 0)):
                return False

            agent.pos = agent.pos % self.depo.grid_size # Wrap around the box

            # Disabled for this task
            # deposit[int(agent.pos[0])][int(agent.pos[1])] += self.params["agent_deposit"]

        return True # TODO: for rendering

    """
        Return a similarity dictionary, {"word": agent_count}

        ind: index for the point
    """
    def run_similarity(self, ind, step_num=500, lifespan=100, threshold=1.0):
        # Spawn all agents at a specific point
        self.initialize_agents(spawn_at=self.depo.point_coord[ind]) 
        ptinfo = self.depo.point_info[ind]

        similarity_dic = {}
        for info in self.depo.point_info:
            similarity_dic[info] = 0

        for step in range(step_num):
            # Count the agents passing thru the points
            for i, info in enumerate(self.depo.point_info):
                if (ptinfo == info):
                    continue

                for agent in self.agents:
                    if (spatial.distance.euclidean(agent.pos,
                                                 self.depo.point_coord[i])
                            < threshold):
                        similarity_dic[info] += 1

            self.step()

            # respawn at step
            if (step % lifespan == 0):
                for agent in self.agents:
                    agent.pos = self.depo.point_coord[ind]

        return similarity_dic

    def run(self, step_num=500, return_trace=False):
        self.initialize_agents()

        # Organization: First dimension: agents, second dimension: time steps
        x_t = [np.array([])] * len(self.agents)
        y_t = [np.array([])] * len(self.agents)
        z_t = [np.array([])] * len(self.agents)

        for step in range(step_num):

            if (return_trace):
                for i, a in enumerate(self.agents):
                    x_t[i] = np.append(x_t[i], a.pos[0])
                    y_t[i] = np.append(y_t[i], a.pos[1])
                    z_t[i] = np.append(z_t[i], a.pos[2])

            #print(self.depo.get_deposit(self.agents[100].pos))
            #print(self.agents[100].pos)
            #print(self.agents[100].get_sphere_coord(), end='\n\n')
            #print(np.linalg.norm(self.agents[100].dir))
            #print(self.agents[100].get_sphere_coord()[1:3])

            if (not self.step()):
                print("step: " + str(step))
                break

        #print(x_t)
        #print(y_t)
        #print(z_t)
        return x_t, y_t, z_t