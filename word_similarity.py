import numpy as np
from scipy import spatial
from scipy.stats import spearmanr

from simulation import SlimeSimulation
from deposit_reader import DATA_PATH, Deposit

"""
    A class for storing simlarity ranks
"""
class SimilarityRank:
    def __init__(self, limit=20):
        self.ranks = []
        self.limit = limit
        self.if_delta = False
    
    def insert(self, similarity, info, smaller_is_better=False):
        inserted = False

        for i, r in enumerate(self.ranks):
            if ((not smaller_is_better and r['similarity'] < similarity) or\
                (smaller_is_better and r['similarity'] > similarity)):

                self.ranks.insert(i, {'info': info, 'similarity': similarity})
                inserted = True
                break

        if (not inserted and len(self.ranks) < self.limit):
            self.ranks.append({'info': info, 'similarity': similarity})

        if (len(self.ranks) > self.limit):
            self.ranks = self.ranks[:self.limit]

    """
        Find similarity difference between the two SimilarityRank classes
    """
    def find_delta(self, rank2, one_sided=False):
        if (self.limit != rank2.limit):
            print("find_delta: WARNING: limits are different")

        delta = SimilarityRank(self.limit)
        delta.if_delta = True

        for i1, r1 in enumerate(self.ranks):
            entry = {'info': r1['info'],'i1': i1, 'i2': -1, 's1': r1['similarity'], 's2': -1}
            diff = -1
            found_in_r2 = False

            for i2, r2 in enumerate(rank2.ranks):
                if (r2['info'] == r1['info']):
                    diff = abs(i1 - i2)
                    entry['i2'] = i2
                    entry['s2'] = r2['similarity']
                    found_in_r2 = True
                    break

            if (not found_in_r2):
                diff = self.limit - i1

            delta.insert(diff, entry)

        # Find entries in r2 but not in r1
        if (not one_sided):
            for i2, r2 in enumerate(rank2.ranks):
                entry = {'info': r2['info'],'i1': -1, 'i2': i2, 's1': -1, 's2': r2['similarity']}
                diff = -1
                found_in_r1 = False

                for i1, r1 in enumerate(self.ranks):
                    if (r2['info'] == r1['info']):
                        found_in_r1 = True
                        break

                if (not found_in_r1):
                    diff = rank2.limit - i2
                    delta.insert(diff, entry)
            
        # Change 'similarity' to 'difference'
        for entry in delta.ranks:
            entry['difference'] = entry['similarity']
            del entry['similarity']

        return delta

    """
        Return spearman ranking correlation p value

        remove_POS: for words the format is "word_POS", this will remove the POS tag
        sync_both: syncing both sides of the ranking
    """
    def find_spearman(self, rank2, remove_POS = False, sync_both = True):
        lst1, lst2 = [], []
        lst1f, lst2f = [], []

        for item in self.ranks:
            info = item['info']

            if remove_POS:
                info = info.split('_')[0]

            lst1.append(info)

        for item in rank2.ranks:
            info = item['info']

            if remove_POS:
                info = info.split('_')[0]

            lst2.append(info)

        if (sync_both):
            for a in lst1:
                if a in lst2:
                    lst1f.append(a)
            for a in lst2:
                if a in lst1:
                    lst2f.append(a)

            return spearmanr(lst1f, lst2f)

        return spearmanr(lst1, lst2)    

    def keys(self, similarity=False, no_zero=False):
        output = []

        for a in self.ranks:
            if no_zero and a['similarity'] == 0:
                continue

            if similarity:
                output.append((a['info'], a['similarity']))
            else:
                output.append(a['info'])
        
        return output

    def display(self):
        if (not self.if_delta):
            print("Similarity Ranking:")
            for i, r in enumerate(self.ranks):
                print(str(i) + ": " + str(r['info']) + "\nsimilarity: " + str(r["similarity"]) + "\n--------------------------------")
        else:
            print("Difference of Similarity with limit of {}: ".format(str(self.limit)))
            for i, r in enumerate(self.ranks):
                print("{}\nDifference: {}, Rank1: {}, Rank2: {}, Sim1: {}, Sim2: {}\n--------------------------------".format(
                        r['info']['info'],\
                            r['difference'],\
                            r['info']['i1'],\
                            r['info']['i2'],\
                            r['info']['s1'],\
                            r['info']['s2']))

    def writefile(self, name, method):
        with open(name + "_" + method + ".txt", 'w+', encoding='utf-8') as f:
            if (not self.if_delta):
                #f.write("Similarity Ranking:\n")
                for i, r in enumerate(self.ranks):
                    f.write(str(i) + ": " + str(r['info']) + "\nsimilarity: " + str(r["similarity"]) + "\n--------------------------------\n")
            else:
                f.write("Difference of Similarity with limit of {}: \n".format(str(self.limit)))
                for i, r in enumerate(self.ranks):
                    f.write("{}\nDifference: {}, Rank1: {}, Rank2: {}, Sim1: {}, Sim2: {}\n--------------------------------\n".format(
                            r['info']['info'],\
                                r['difference'],\
                                r['info']['i1'],\
                                r['info']['i2'],\
                                r['info']['s1'],\
                                r['info']['s2']))

"""
    Given a ranked list, generate rank2 until all items in rank1 is found
    Then calculate their difference
"""
def find_exhaustive_difference(depo, pt, rank1, rank2_measure, initial_limit=100, limit_incre=100, return_rank2=False):
    limit = initial_limit
    found_all_item = False
    rank2 = None

    while(not found_all_item):
        found_all_item = True
        limit += limit_incre
        rank2 = find_similarity_rank(depo, pt, num_words=limit, measure=rank2_measure)

        for item1 in rank1.ranks:
            found_one_item = False

            for item2 in rank2.ranks:
                if (item1['info'] == item2['info']):
                    found_one_item = True
                    break

            if (not found_one_item):
                found_all_item = False
                break
    
    if return_rank2:
        return rank2, rank1.find_delta(rank2, one_sided=True)

    return rank1.find_delta(rank2, one_sided=True)


"""
    Given a point (either word text or coordinate), prints out words most similar to it.
    num_words gives the number of words displayed
    slime_lifespan is the lifespan of the agents
"""
def find_similarity_rank(depo, pt, num_words=20, measure="cosine",\
                     slime_steps=500,  slime_lifespan=500, threshold=1.0,\
                     verbose=False, stop_when_explored=False,\
                     ignore_point=False):
    ind = -1

    # Look for word index
    if (type(pt) is np.ndarray or type(pt) is list):
        for i, coord in enumerate(depo.point_coord):
            if (np.all(np.isclose(pt, coord))):
                ind = i
                break
    elif (type(pt) is str):
        for i, coord in enumerate(depo.point_info):
            if (pt == coord):
                ind = i
                break
    else:
        print("find_similarity_rank: pt type is not supported")
        return

    # See if measure is supported
    if (measure not in ['cosine', 'euclidean', 'slime_mold']):
        print("find_similarity_rank: unknown similarity measure")
        return

    # Check if found
    if (ind == -1):
        print("find_similarity_rank: pt given is not found")
        return
    
    if (verbose):
        print("Finding similarity for word: " + depo.point_info[ind])
        print("Coordinate: " + str(depo.point_coord[ind]))

    # Start finding ranks
    ranks = SimilarityRank(num_words) # Descending order
    pt = depo.point_coord[ind]
    origin = np.mean(depo.point_coord, axis=0)

    for i, coord in enumerate(depo.point_coord):
        if (i == ind):
            continue

        word = depo.point_info[i]
        similarity = -1

        if (measure is 'cosine'):
            similarity = cosine_similarity(coord, pt, origin)
            ranks.insert(similarity, word)
        elif (measure is 'euclidean'):
            similarity = spatial.distance.euclidean(coord, pt)
            ranks.insert(similarity, word, smaller_is_better=True)

    if (measure is 'slime_mold'):
        sim = SlimeSimulation(depo)
        similarity_list = sim.run_similarity(ind,
                                    step_num=slime_steps,
                                    lifespan=slime_lifespan,
                                    threshold=threshold,
                                    stop_when_explored=stop_when_explored)
        for k in similarity_list.keys():
            ranks.insert(similarity_list[k], k)

    return ranks

"""
    Find cosine similarity
"""
def cosine_similarity(pt1, pt2, origin = np.array([0, 0, 0])):
    pt1 = pt1 - origin
    pt2 = pt2 - origin

    return 1 - spatial.distance.cosine(pt1, pt2)
