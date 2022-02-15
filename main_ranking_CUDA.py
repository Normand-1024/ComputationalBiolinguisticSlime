import sys
import os
import math
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from scipy.spatial import distance
from mpl_toolkits.mplot3d import Axes3D
from word_similarity import SimilarityRank, find_similarity_rank, find_exhaustive_difference

import util

from simulation import SlimeSimulation
from deposit_reader import DATA_PATH, Deposit
from simulation_CUDA import get_agent_array, CUDA_slime

total_step = 1500
reset_step = 1500
start_token = 120592                                  # EDIT THIS TO      PICK UP WHERE YOU LEFT OFF
lowest_node_weight = 0.6                            # EDIT THIS TO      SET LOWEST NODE WEIGHT THRESHOLD FOR FILE GENERATION
default_node = 0                                    # EDIT THIS TO      SET THE DEFAULT NODE ID WHEN LOADED
path = "word_similarity_result\\graph\\master_global_2\\"   # EDIT THIS TO      CHANGE THE RESULT FOLDER PATH

depo = Deposit(DATA_PATH)
depo.params["num_agent"] = 3000
sim = SlimeSimulation(depo)
deposit = np.squeeze(depo.deposit.astype(np.float32), axis=3)

grid_res = depo.grid_res.astype(np.int32) # unadjusted, dimension of the grid in array coordinate
grid_size = depo.grid_size.astype(np.float32) # unadjusted, dimension of the grid in world coordinate
parameter = np.array([depo.params["sense_angle"],\
            depo.params["sense_distance"],\
            depo.params["sharpness"],\
            depo.params["move_angle"],\
            depo.params["move_distance"]]).astype(np.float32)

#
#   If any argv is entered, generate meta file of the folder
#       I do this because I don't want another file
#
if (len(sys.argv) > 1):
    with open(path + "meta-main.txt", 'w+', encoding='utf-8') as f:

        f.write(str(default_node) + '\n') # The node displayed upon load
        numbers = []

        # Write the list of ids available, seperated by white space
        #       Assuming that all file names only contain its id other than extension
        for filename in os.listdir(os.getcwd() + "\\" + path):
            if (filename != "meta-main.txt" and filename != "full_data"):
                numbers.append(int(filename.split('.')[0]))

        numbers.sort()
        
        for i in numbers:
            f.write(str(i) + ' ' + depo.point_info[i] + '\n')
    exit()

################################
################################

# 
#   Main loop to generate
#
for i in range(start_token, len(depo.point_coord)):
    if (depo.point_weight[i] >= lowest_node_weight):
        spawn = depo.point_coord[i]
        print("doing: " + str(i) + ": " + depo.point_info[i], end='')
        start_time = time.time()

        sim.initialize_agents(spawn_at=spawn)
        agent_array = get_agent_array(sim.agents)
        
        slime_sim = CUDA_slime(deposit, agent_array,\
                    depo.point_coord, depo.point_info, depo.point_weight,\
                    grid_res, grid_size,\
                    parameter)

        slime_sim.transferAgentTraceTex()

        for j in range(total_step):
            slime_sim.step(add_trace=False)

            if j % reset_step == 0:
                for agent in slime_sim.agent_array:
                    agent[0][0] = spawn[0]
                    agent[0][1] = spawn[1]
                    agent[0][2] = spawn[2]
        
        slime_sim.retrieveAgentTraceTex()
        print(", %s" % (time.time() - start_time), end="")

        slime_sim.generateSimilarity()
        print(", %s" % (time.time() - start_time), end="")

        slime_sim.ranking_threshold = 1000
        ranking = slime_sim.getSimilarityRank()
        print(", %s" % (time.time() - start_time), end="")

        # The file name: Simply the id - its position on the array
        filename = path + str(i) + ".txt"

        # 
        #   Similarity format: "info \n (index, similarity, node weight, coordinate)"   
        #
        with open(filename, mode='w+', encoding='utf-8') as f:
            for similarity in ranking: # Format: i2, similarity (connect from i to i2)
                f.write(str(similarity[0]) + ', ' + str(similarity[1]) + '\n')

            """ 
            FOR THE OLD OUTPUT METHOD, TAKES A LOT OF SPACE

            for similarity in ranking:#i, similarity in enumerate(slime_sim.similarity_rank):
                f.write(depo.point_info[similarity[0]] + '\n' +\
                    str(similarity) + '\n')
            """
        print(", time: %s" % (time.time() - start_time))

        slime_sim.clearGPUMemory()


# 
# ABANDONED CODE
#

""" 
filename = "".join(x for x in depo.point_info[i] if x.isalnum())
if len(filename) > 20:
    filename = path + str(i) + "-" + filename[:20]+".txt"
else:
    filename = path + str(i) + "-" + filename +".txt" 
"""