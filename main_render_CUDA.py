import math
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from mpl_toolkits.mplot3d import Axes3D
from word_similarity import SimilarityRank, find_similarity_rank, find_exhaustive_difference

import util

from simulation import SlimeSimulation
from deposit_reader import DATA_PATH, Deposit
from simulation_CUDA import get_agent_array, CUDA_slime

def find_nearest_point(depo, pt_coord):
    output_coord = None
    dist = None
    for p in depo.point_coord:
        if not dist or dist > distance.euclidean(pt_coord, p):
            output_coord = p
            dist = distance.euclidean(pt_coord, p)
    return output_coord

depo = Deposit(DATA_PATH)
depo.params["num_agent"] = 300
sim = SlimeSimulation(depo)

spawn_point = depo.point_coord[0]
filename = "classs.png"
start_time = time.time()

"""
#=========== THIS IS ALSO IRRELEVANT A LOT OF TIMES =====
pt = np.array([383, 225, 214])
# left cluster [383, 225, 214]
#top right cluster [144, 139, 149]
#bottom right cluster [191, 245, 274]
spawn_point = find_nearest_point(depo, pt)
# ===========================================
"""

# class_NOUN is a path point, Medium_PROPN\n is a cluster point
# Medium_PROPN, class_NOUN
for i, pti in enumerate(depo.point_info):
    if pti == 'class_NOUN':
        print("Found at {}".format(i))
        spawn_point = depo.point_coord[i]
        break

slic_z = spawn_point[2]#depo.grid_size[2] / 2
margin = 10
slice_back = slic_z - margin
slice_front = slic_z + margin

"""
x_t, y_t, z_t, similarity_dic = sim.run_with_similarity(\
    step_num=500,  lifespan=500,\
     spawn_at = spawn_point)
"""

#
# CUDA Slime Simulation Setup
#
sim.initialize_agents(num_agents=1000, spawn_at=spawn_point)
grid_res = depo.grid_res.astype(np.int32) # unadjusted, dimension of the grid in array coordinate
grid_size = depo.grid_size.astype(np.float32) # unadjusted, dimension of the grid in world coordinate
parameter = np.array([depo.params["sense_angle"],\
             depo.params["sense_distance"],\
             depo.params["sharpness"],\
             depo.params["move_angle"],\
             depo.params["move_distance"]]).astype(np.float32)

agent_array = get_agent_array(sim.agents)
deposit = np.squeeze(depo.deposit.astype(np.float32), axis=3)

# 
# CUDA Slime Simulation Main
x_t, y_t, z_t = [], [], []
slime_sim = CUDA_slime(deposit, agent_array,\
            grid_res, grid_size,\
            parameter)

total_step = 500
for i in range(total_step):
    #print(i, ' / ', total_step, end='\r')
    slime_sim.step()

    for agent in slime_sim.agent_array:
        x_t += [agent[0][0]]
        y_t += [agent[0][1]]
        z_t += [agent[0][2]]
        
#
# Set up figure
#
fig = plt.figure(figsize=(10,10))
x_flat = np.array(x_t).flatten()
y_flat = np.array(y_t).flatten()
x_min, x_max = np.amin(x_flat), np.amax(x_flat)
y_min, y_max =  np.amin(y_flat), np.amax(y_flat)
max_len = max(x_max - x_min, y_max - y_min)
axes = fig.add_subplot(111)

# Set view
axes.set_xlim(x_min - 0, x_min + max_len + 0)
axes.set_ylim(y_min - 0, y_min + max_len + 0)
axes.set_xlabel('x')
axes.set_ylabel('y')
print("{}, {}, {}".format(x_min, y_min, max_len))

#
# Plotting agent travel traces
#
axes.scatter(x_t, y_t, alpha=0.02, marker='.',\
        color='gray', s=1) # Traces

x_undis, y_undis = [], []
for coor in depo.point_coord:
    if coor[2] >= slice_back and coor[2] <= slice_front:
        x_undis.append(coor[0])
        y_undis.append(coor[1])
axes.scatter(x_undis, y_undis, alpha = 0.08,\
    marker = 'o',color='green', s=2) # Word tokens


print(depo.grid_size)
print("Time Elapsed: " + str(time.time() - start_time))
#plt.savefig(filename, dpi=300)
plt.show()

exit()
