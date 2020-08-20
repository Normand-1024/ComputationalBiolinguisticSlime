import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import util

"""
Simulation sanity check
"""
from simulation import SlimeSimulation
from deposit_reader import DATA_PATH, Deposit
depo = Deposit(DATA_PATH)
sim = SlimeSimulation(depo)

for n in range(500):
    x_a, y_a, z_a = sim.run(step_num=8000, return_trace=True)

    # For plotting travel paths

    fig = plt.figure()
    axes = fig.add_subplot(111)#, projection='3d')

    for i, _ in enumerate(x_a):
        plt.plot(x_a[i], y_a[i], 'b-', alpha=0.01, rasterized=True)

    plt.xlim(0, depo.grid_size[0])
    plt.ylim(0, depo.grid_size[0])
    plt.savefig('plot_png/slime' + str(n) +'.png', transparent=True, dpi=500)

    plt.clf()
#plt.show()

exit()

"""
TEST CODE
"""
from deposit_reader import DATA_PATH, Deposit
from word_similarity import SimilarityRank, find_similarity_rank, find_exhaustive_difference

depo = Deposit(DATA_PATH)
pt = depo.point_coord[0]
slime_r = find_similarity_rank(depo, pt, 30, 'slime_mold', slime_steps=2000,  slime_lifespan=600)
slime_r.display()
diff_cos_r = find_exhaustive_difference(depo, pt, slime_r, 'cosine')
diff_eucd_r = find_exhaustive_difference(depo, pt, slime_r, 'euclidean')
diff_cos_r.display()
diff_eucd_r.display()
exit()

cos_r = find_similarity_rank(depo, pt, 20, 'cosine')
eucd_r = find_similarity_rank(depo, pt, 20, 'euclidean')
diff_r = find_exhaustive_difference(depo, pt, cos_r, 'euclidean')#cos_r.find_delta(eucd_r)

diff_r.display()

exit()

# For plotting deposit at agents
"""
deposit_values = []
for a in sim.agents:
    deposit_values.append(depo.get_deposit(a.pos))
plt.hist(deposit_values)
plt.show()
"""

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

# For plotting agent points
"""
slic_z = 100
margin = 5
plot_pt = np.array([])
for a in sim.agents:
    if (a.pos[2] <= slic_z + margin and\
        a.pos[2] >= slic_z - margin):
        plot_pt = np.append(plot_pt, np.array(a.pos[0], a.pos[1]))
plt.plot(plot_pt, 'o')
plt.show()
"""

exit()
