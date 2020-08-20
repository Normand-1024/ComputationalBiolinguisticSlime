import numpy as np
from scipy import spatial

import real_dataset.SimLex999.readSimLex as rsl
from deposit_reader import DATA_PATH, Deposit
from word_similarity import SimilarityRank, find_similarity_rank, find_exhaustive_difference, cosine_similarity


depo = Deposit(DATA_PATH)
depo.params['num_agent'] = 300

simlex = rsl.readSimLex("real_dataset/SimLex999/SimLex-999.txt")

cnt = 0
for i, w in enumerate(depo.point_info):
    word, pos = w.split('_')[:2]

    # 
    #   1. Filter the to see if all words are in embedding data
    #

    if pos not in ['NOUN', 'VERB', 'ADJ']:
        continue

    if not simlex.simlexdict.get(word):
        continue

    if len(simlex.simlexdict.get(word).keys()) < 2:
        continue

    if pos[0] != list(simlex.simlexdict.get(word).values())[0][0]:
        continue

    simlex_r = simlex.get_rank(word) # Obtain simlex ranking here

    # Check if all ranking words exist in point_info
    can_words_be_found = True
    word_indices = []
    for w2 in simlex_r.keys():
        try:
            ind = depo.point_info.index(w2)
            word_indices.append(ind)
        except ValueError:
            can_words_be_found = False
            break

    if not can_words_be_found:
        continue

    # 
    #   2. Find Rankings
    #

    cos_r = find_similarity_rank(depo, w, 30, 'cosine')
    eucd_r = find_similarity_rank(depo, w, 30, 'euclidean')
    slime_r = find_similarity_rank(depo, w, 30,'slime_mold',\
         slime_steps=20,  slime_lifespan=500, threshold = 10.0)


    #
    #   3. Generate data for printing
    #
    simlex_rl = simlex_r.keys(similarity=False)
    cos_rl = cos_r.keys(similarity=False)
    eucd_rl = eucd_r.keys(similarity=False)
    slime_rl = slime_r.keys(similarity=False, no_zero=True)
    hit_count = 0
    hit_list = []

    for h in simlex_rl:
        try:
            ind = slime_rl.index(h)
            hit_count += 1
            hit_list.append(ind + 1)
        except ValueError:
            hit_list.append(0)

    avg_pos = 0
    if hit_count > 0:
        avg_pos = sum(hit_list) / hit_count

    #
    #   4. Print result
    #
    cnt += 1

    print("### {} ###".format(w))
    print("SimlexRank: {}".format(simlex_rl))
    print("CosRank: {}".format(cos_rl))
    print("EucdRank: {}".format(eucd_rl))
    print("SlimeRank: {}".format(slime_rl))
    print("SlimeHitCount: {}".format(hit_count))
    print("SlimeHitList: {}".format(hit_list))
    print("SlimeAvgPos: {}".format(avg_pos))
    print("=============================")

    exit()

"""
=====================================================
 ABANDONED ZONE AHEAD
=====================================================
"""


def percise_cosine(depo, i, word_list):
    w0_coord, w0_info = depo.point_coord[i], depo.point_info[i]
    origin = np.mean(depo.point_coord, axis=0)

    r1 = SimilarityRank(limit=20)

    for w in word_list:
        ind = depo.point_info.index(w)
        coor = depo.point_coord[ind]
        sim = cosine_similarity(coor, w0_coord, origin)
        r1.insert(sim, w)

    return r1

def percise_euclidean(depo, i, word_list):
    w0_coord, w0_info = depo.point_coord[i], depo.point_info[i]

    r1 = SimilarityRank(limit=20)

    for w in word_list:
        ind = depo.point_info.index(w)
        coor = depo.point_coord[ind]
        sim = spatial.distance.euclidean(coor, w0_coord)
        r1.insert(sim, w, smaller_is_better=True)

    return r1

def percise_slime(depo, i, index_list, max_steps = 2000, threshold=1.0):
    w0_coord, w0_info = depo.point_coord[i], depo.point_info[i]

    r1 = SimilarityRank(limit=20)

    point_info0, point_coord0 = depo.point_info, depo.point_coord

    point_info1, point_coord1 = [point_info0[i]], [point_coord0[i]]
    for ind in index_list:
        point_info1.append(point_info0[ind])
        point_coord1.append(point_coord0[ind])

    depo.point_info, depo.point_coord = point_info1, point_coord1
    depo.point_grid, depo.sorted_point = depo.preprocess_ptdata(downsample=2)

    output = find_similarity_rank(depo, w0_coord, measure='slime_mold',\
         slime_steps = max_steps, slime_lifespan = max_steps, stop_when_explored=True)

    depo.point_info, depo.point_coord = point_info0, point_coord0

    return output

depo = Deposit(DATA_PATH)
depo.params['num_agent'] = 300

simlex = rsl.readSimLex("real_dataset/SimLex999/SimLex-999.txt")
#rank = simlex.get_rank('old')
#rank.display()

#print(rank.find_spearman(simlex.get_rank('old')))

spearmans_cos, spearmans_eucd, spearmans_slime = [], [], []
cnt=0

for i, w in enumerate(depo.point_info):
    word, pos = w.split('_')[:2]

    if pos not in ['NOUN', 'VERB', 'ADJ']:
        continue

    if not simlex.simlexdict.get(word):
        continue

    if len(simlex.simlexdict.get(word).keys()) < 2:
        continue

    if pos[0] != list(simlex.simlexdict.get(word).values())[0][0]:
        continue

    simlexrank = simlex.get_rank(word)

    # Check if all ranking words exist in point_info
    can_words_be_found = True
    word_indices = []
    for w2 in simlexrank.keys():
        try:
            ind = depo.point_info.index(w2)
            word_indices.append(ind)
        except ValueError:
            can_words_be_found = False
            break

    if not can_words_be_found:
        continue

    # !!!!!!!!!!!!!!!!!!!!    
    if cnt >= 100:
        break
    elif cnt < 50:
        cnt += 1
        continue
    else:
        cnt += 1
    # !!!!!!!!!!!!!!!!!!!!

    # Actually start doing stuff
    w_coord = depo.point_coord[i]

    #cos_rank = percise_cosine(depo, i, [k['info'] for k in simlexrank.ranks])
    #eucd_rank = percise_euclidean(depo, i, [k['info'] for k in simlexrank.ranks])
    slime_rank = percise_slime(depo, i, word_indices)
    
    # Calculate Spearman
    #spearmans_cos.append(simlexrank.find_spearman(cos_rank))
    #spearmans_eucd.append(simlexrank.find_spearman(eucd_rank))
    spearmans_slime.append(simlexrank.find_spearman(slime_rank))

    # Print them out:
    print(">>>> {} <<<<<".format(w))
    print("Simlex: {}".format(simlexrank.keys(similarity=True)))
    #print("Cos: {}".format(cos_rank.keys(similarity=True)))
    #print("Eucd: {}".format(eucd_rank.keys(similarity=True)))
    print("Slime: {}".format(slime_rank.keys(similarity=True)))
    print("===============")

# Display Spearman
#print(spearmans_cos)
#print(spearmans_eucd)
print(spearmans_slime)
pvalue_spearman_cos = [a[1] for a in spearmans_cos]
pvalue_spearman_eucd = [a[1] for a in spearmans_eucd]
pvalue_spearman_slime = [a[1] for a in spearmans_slime]

for i, a in enumerate(pvalue_spearman_cos):
    if np.isnan(a):
        if spearmans_cos[i][0] > 0:
            pvalue_spearman_cos[i] = 1
        else:
            pvalue_spearman_cos[i] = 0
for i, a in enumerate(pvalue_spearman_eucd):
    if np.isnan(a):
        if spearmans_eucd[i][0] > 0:
            pvalue_spearman_eucd[i] = 1
        else:
            pvalue_spearman_eucd[i] = 0
for i, a in enumerate(pvalue_spearman_slime):
    if np.isnan(a):
        if spearmans_slime[i][0] > 0:
            pvalue_spearman_slime[i] = 1
        else:
            pvalue_spearman_slime[i] = 0

#print("Cos spearman mean: {}".format(np.mean(pvalue_spearman_cos)))
#print("Eucd spearman mean: {}".format(np.mean(pvalue_spearman_eucd)))
print("Slime spearman mean: {}".format(np.mean(pvalue_spearman_slime)))