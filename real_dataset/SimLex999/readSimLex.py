from word_similarity import SimilarityRank

POS, SIMLEX = 0, 1

class readSimLex:
    def __init__(self, filename):
        self.simlexdict = {}

        with open(filename,'r', encoding='utf-8') as simlex:
            firstln = True

            for ln in simlex:
                # Skip the first line
                if firstln:
                    firstln = False
                    continue

                ln = ln.split('\t')
                
                if not self.simlexdict.get(ln[0]):
                    self.simlexdict[ln[0]] = {}

                if self.simlexdict.get(ln[0]).get(ln[1]):
                    print("ERROR: ENTRIES REPEATS: {}, {}".format(ln[0], ln[1]))
                    exit()

                self.simlexdict[ln[0]][ln[1]] = ln[2:]

    def get_rank(self, word1, pos=True):
        if (not self.simlexdict.get(word1)):
            print("ERROR: get_rank: word1 not found")
            exit()

        output = SimilarityRank(limit = 1000)

        for k in self.simlexdict[word1].keys():
            info = k
            if pos:
                pos = "NOUN"

                if self.simlexdict[word1][k][POS] == "V":
                    pos = "VERB"
                elif self.simlexdict[word1][k][POS] == "A":
                    pos = "ADJ"

                info = info + '_' + pos

            similarity = float(self.simlexdict[word1][k][SIMLEX])
            output.insert(similarity, info)

        return output