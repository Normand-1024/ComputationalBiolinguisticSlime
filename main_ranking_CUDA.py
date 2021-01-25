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
start_token = 0
path = "word_similarity_result\\graph\\TNG-100\\"

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
#   Decide whether to expansively generate networks or simply looping thru
#
expansive_graph_building = False

if expansive_graph_building:
    queue, discovered = deque(), [] # Generating a graph based on an origin point
    queue.append(0)
    queue.append(1)
    while len(queue) is not 0:
        i = queue.popleft()
        spawn = depo.point_coord[i]

        print("doing: " + str(i) + ": " + depo.point_info[i])
        print("currentStatus: ", queue, " || ", discovered)

        sim.initialize_agents(spawn_at=spawn)
        agent_array = get_agent_array(sim.agents)
        
        slime_sim = CUDA_slime(deposit, agent_array,\
                    depo.point_coord, depo.point_info, depo.point_weight,\
                    grid_res, grid_size,\
                    parameter)

        for j in range(total_step):
            slime_sim.stepWithoutRecord()

            if j % 500 == 0:
                for agent in slime_sim.agent_array:
                    agent[0][0] = spawn[0]
                    agent[0][1] = spawn[1]
                    agent[0][2] = spawn[2]

        slime_sim.recordTrace()
        slime_sim.generateSimilarity()

        ranking = slime_sim.getSimilarityRank()

        filename = "".join(x for x in depo.point_info[i] if x.isalnum())
        if len(filename) > 20:
            filename = path + str(i) + "-" + filename[:20]+".txt"
        else:
            filename = path + str(i) + "-" + filename +".txt"

        # Push the similarity rankings to the queue and write to file
        with open(filename, 'w+', encoding='utf-8') as f:
            for entry in ranking:
                f.write(depo.point_info[entry[0]] + '\n' +\
                    str(entry[1]) + '\n')
                #if entry[0] not in queue and entry[0] not in discovered:
                #    queue.append(entry[0])
else:
    for i in range(start_token, len(depo.point_coord)):
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

        ranking = slime_sim.getSimilarityRank()
        print(", %s" % (time.time() - start_time), end="")

        filename = "".join(x for x in depo.point_info[i] if x.isalnum())
        if len(filename) > 20:
            filename = path + str(i) + "-" + filename[:20]+".txt"
        else:
            filename = path + str(i) + "-" + filename +".txt"

        # 
        #   Similarity format: "info \n (index, similarity, weight)"   
        #
        with open(filename, 'w+', encoding='utf-8') as f:
            for similarity in ranking:#i, similarity in enumerate(slime_sim.similarity_rank):
                f.write(depo.point_info[similarity[0]] + '\n' +\
                    str(similarity) + '\n')

        print(", time: %s" % (time.time() - start_time))
