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

THE_WORD_IM_LOOKING_FOR = 'love_VERB'

depo = Deposit(DATA_PATH)
idex = 0

# class_NOUN\n is a path point, Medium_PROPN\n is a cluster point
for i, pti in enumerate(depo.point_info):
    if pti == THE_WORD_IM_LOOKING_FOR:
        idex = i
        break

pt = depo.point_coord[idex]

#=========== THIS IS ALSO IRRELEVANT A LOT OF TIMES =====
#pt = np.array([383, 225, 214])
#pt = find_nearest_point(depo, pt)
# ===========================================

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

cos_r = find_similarity_rank(depo, pt, 500, 'cosine')
eucd_r = find_similarity_rank(depo, pt, 500, 'euclidean')
cos_r.writefile(THE_WORD_IM_LOOKING_FOR, 'cosine')
eucd_r.writefile(THE_WORD_IM_LOOKING_FOR, 'euclidean')

print("Cos and Euc finished")

exit()

depo.params['num_agent'] = 300 # For controlling agent numbers
slime_r = find_similarity_rank(depo, pt, 30, 'slime_mold', slime_steps=1500,  slime_lifespan=500)
print("Slime finished")
slime_r.writefile(THE_WORD_IM_LOOKING_FOR, 'slime')
exit()

diff_cos_r = find_exhaustive_difference(depo, pt, slime_r, 'cosine')
diff_eucd_r = find_exhaustive_difference(depo, pt, slime_r, 'euclidean')
diff_cos_r.writefile(THE_WORD_IM_LOOKING_FOR, 'slime-cosine')
diff_eucd_r.writefile(THE_WORD_IM_LOOKING_FOR, 'slime-eucli')

