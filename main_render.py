import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from mpl_toolkits.mplot3d import Axes3D
from word_similarity import SimilarityRank, find_similarity_rank, find_exhaustive_difference

import util

from simulation import SlimeSimulation
from deposit_reader import DATA_PATH, Deposit
depo = Deposit(DATA_PATH)
depo.params["num_agent"] = 300
sim = SlimeSimulation(depo)

def find_nearest_point(depo, pt_coord):
    output_coord = None
    dist = None
    for p in depo.point_coord:
        if not dist or dist > distance.euclidean(pt_coord, p):
            output_coord = p
            dist = distance.euclidean(pt_coord, p)
    return output_coord

spawn_point = depo.point_coord[0]
filename = "classs.png"

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
    if pti == 'class_NOUN?':
        print("Found at {}".format(i))
        spawn_point = depo.point_coord[i]
        break

x_t, y_t, z_t, similarity_dic = sim.run_with_similarity(\
    step_num=1500,  lifespan=500,\
     spawn_at = spawn_point)

slic_z = spawn_point[2]#depo.grid_size[2] / 2
margin = 10
slice_back = slic_z - margin
slice_front = slic_z + margin

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

# Larger viewing
#x_min, y_min, max_len = spawn_point[0] - 70, spawn_point[1] - 70, 140

#axes.set_xlim(0, depo.grid_size[0])
#axes.set_ylim(0, depo.grid_size[1])
# 5
axes.set_xlim(x_min - 0, x_min + max_len + 0)
axes.set_ylim(y_min - 0, y_min + max_len + 0)
axes.set_xlabel('x')
axes.set_ylabel('y')
print("{}, {}, {}".format(x_min, y_min, max_len))

#
# Plotting agent travel traces
#
plottingstyle = 2 # 1: all, 2: slice
if (plottingstyle == 1):
    for a in range(len(x_t)):
        plt.scatter(x_t[a], y_t[a], alpha=0.002, marker='.',\
             color='gray', s=1)
elif (plottingstyle == 2):
    x_plot = np.array([])
    y_plot = np.array([])
    for a in range(len(x_t)):
        for b in range(len(x_t[0])):
            if (z_t[a][b] >= slice_back and z_t[a][b] <= slice_front):
                x_plot = np.append(x_plot, x_t[a][b])
                y_plot = np.append(y_plot, y_t[a][b])
    axes.scatter(x_plot, y_plot, alpha=0.06, marker='.',\
         color='grey', s=2)
            
    # 
    # Plotting discovered dots and undiscovered dots
    #

    x_undis, y_undis = [], []
    for coor in depo.point_coord:
        if coor[2] >= slice_back and coor[2] <= slice_front:
            x_undis.append(coor[0])
            y_undis.append(coor[1])

    #0.02
    axes.scatter(x_undis, y_undis, alpha = 0.08,\
        marker = 'o',color='green', s=2)

    discovered_dots = [i for i in similarity_dic if similarity_dic[i] != 0]
    x_discovered, y_discovered = [], []
    for i in discovered_dots:
        coor = depo.point_coord[i]
        if coor[2] >= slice_back and coor[2] <= slice_front:
            x_discovered.append(depo.point_coord[i][0])
            y_discovered.append(depo.point_coord[i][1])
    #0.05
    axes.scatter(x_discovered, y_discovered, alpha=0.1,\
            marker='.',color='red', s=2)

axes.scatter([spawn_point[0]], spawn_point[1], alpha=1, marker='.',\
        color='blue', s=30)


print(depo.grid_size)
plt.savefig(filename, dpi=300)
#plt.show()

exit()

# =================================================================
# =================================================================

# For plotting travel paths
fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')
x1, y1, z1 = x_t[10], y_t[10], z_t[10]
x2, y2, z2 = x_t[11], y_t[11], z_t[11]
x3, y3, z3 = x_t[12], y_t[12], z_t[12]
plt.plot(x1, y1, z1, '-')
plt.plot(x2, y2, z2, '-')
plt.plot(x3, y3, z3, '-')
plt.xlim(0, depo.grid_size[0])
plt.ylim(0, depo.grid_size[1])
plt.show()

# For plotting polar coordinate
theta, phi = np.array([]), np.array([])
for i, a in enumerate(sim.agents):
    r, thetaa, phia = a.get_sphere_coord()
    theta = np.append(theta, thetaa)
    phi = np.append(phi, phia)

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
axs[0].hist(theta)
axs[1].hist(phi)
plt.show()

exit()


# For plotting deposit at agents
deposit_values = []
for a in sim.agents:
    deposit_values.append(depo.get_deposit(a.pos))
plt.hist(deposit_values)
plt.show()

exit()

"""
CODE FOR DISPLAYING SLICE
"""
from deposit_reader import DATA_PATH, Deposit

GAMMA = 0.2
FIGSIZE = 10.0
z = 150
margin = 10
immargin = 10
depo = Deposit(DATA_PATH)
voxels_u8 = depo.deposit.astype(np.uint8)

the_pt = []

# class_NOUN is a path point, Medium_PROPN\n is a cluster point
# Medium_PROPN
for i, pti in enumerate(depo.point_info):
    if pti == 'class_NOUN':
        z = math.floor(depo.point_coord[i][0])
        print("Found at {}".format(i))
        the_pt = [depo.point_coord[i][2] * depo.grid_ratio, depo.point_coord[i][1] * depo.grid_ratio]
        break

# For finding a bunch of points in the cluster
"""
for i, ptc in enumerate(depo.point_coord):
    if (ptc[0] >= z - margin and ptc[0] <= z + margin and ptc[2] >= 550 / depo.grid_ratio and ptc[2] <= 650  / depo.grid_ratio and ptc[1] >= 300 / depo.grid_ratio and ptc[1] <= 350 / depo.grid_ratio):
        print("IN CLUSTER: {}, {}".format(ptc, depo.point_info[i]))
"""

pt_array_x = []
pt_array_y = []
for pt in depo.point_coord:
    if (pt[0] >= z - margin and pt[0] <= z + margin):
        pt_array_x.append(pt[2] * depo.grid_ratio)
        pt_array_y.append(pt[1] * depo.grid_ratio)

lowerb = math.floor((z-immargin) * depo.grid_ratio)
upperb = math.floor((z+immargin) * depo.grid_ratio)
print([lowerb, upperb])
voxels_u8 = voxels_u8[:,:,lowerb : upperb,0] ** GAMMA
voxels_u8 = np.sum(voxels_u8, axis=2)
plt.figure(figsize = (FIGSIZE, FIGSIZE))
plt.imshow(voxels_u8, origin='lower')
plt.scatter(pt_array_y, pt_array_x, s=1, color='b', alpha=0.2)
plt.scatter([the_pt[1]], [the_pt[0]], s=5, color='red')
plt.xlim(the_pt[1] - 150, the_pt[1] + 150)
plt.ylim(the_pt[0] - 150, the_pt[0] + 150)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

exit()

#===============================================