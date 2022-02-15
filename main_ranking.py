import math
import random
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import util

def find_nearest_point(depo, pt_coord):
    output_coord = None
    dist = None
    for p in depo.point_coord:
        if not dist or dist > distance.euclidean(pt_coord, p):
            output_coord = p
            dist = distance.euclidean(pt_coord, p)
    return output_coord

"""
TEST CODE FOR SIMILARITY
"""
from deposit_reader import DATA_PATH, Deposit
from word_similarity import SimilarityRank, find_similarity_rank, find_exhaustive_difference

#"run_VERB", "love_VERB", "wind_NOUN"
THE_WORD_IM_LOOKING_FOR = 'wind_NOUN' 

depo = Deposit(DATA_PATH)
idex = 0

# class_NOUN\n is a path point, Medium_PROPN\n is a cluster point
for i, pti in enumerate(depo.point_info):
    if pti == THE_WORD_IM_LOOKING_FOR:
        idex = i
        print("index is: " + str(idex))
        break

#exit()

pt = depo.point_coord[idex]

#=========== THIS IS ALSO IRRELEVANT A LOT OF TIMES =====
#pt = np.array([383, 225, 214])
#pt = find_nearest_point(depo, pt)
# ===========================================


#=========== GENERATE DATA FROM VIS RESULT ================
# 1928 - love_VERB, 107 - run_VERB, 1285 - wind_NOUN
if (True):
    with open('1928.txt', 'r') as readf:
        the_word = depo.point_info[1928]
        with open(the_word + "_slimmmeee.txt", 'w+', encoding='UTF-8') as writef:
            for ln in readf:
                print(ln)
                ln = ln.split(',')
                print(ln)
                the_other_word = depo.point_info[eval(ln[0])]
                writef.write(the_other_word + ',' + ln[1])

    exit()
#===============================================


#========== GENERATE COS AND EUCD SMILARITY ============

cos_r = find_similarity_rank(depo, pt, 5000, 'cosine')
eucd_r = find_similarity_rank(depo, pt, 5000, 'euclidean')
cos_r.writefile(THE_WORD_IM_LOOKING_FOR, 'cosine')
eucd_r.writefile(THE_WORD_IM_LOOKING_FOR, 'euclidean')

print("Cos and Euc finished")

exit()

#========================================================


# ================= delete this =================
"""
for i, coord in enumerate(depo.point_coord):
    if coord[0] >= 115 and coord[0] <= 120 and\
        coord[1] <= 105 and coord[1] >= 100 and\
        coord[2] <= depo.point_coord[idex][2] + 5 and\
        coord[2] >= depo.point_coord[idex][2] - 5:
        print(coord)
        print(depo.point_info[i])
        exit()
"""
# ================= =============================

