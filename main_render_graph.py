import os
import matplotlib.pyplot as plt
import networkx as nx

from networkx.algorithms import moral
from deposit_reader import DATA_PATH, Deposit

directory = "word_similarity_result\\graph\\TNG-100\\"
filename = "562-562.txt"
readfile = directory + filename
count_connection = False
rendering_graph = True

#=========Count the connection============

if count_connection:
    connectivity_count = []
    count = 0
    filenum = os.listdir(directory)
    for filename in os.listdir(directory):
        print(str(count) + "/" + str(filenum), end='\r', flush=True)
        count += 1
        name = filename.split('-')[1].split('.')[0]
        number = len(open(directory + filename, 'r', encoding='utf-8').readlines()) / 2
        connectivity_count.append((name, number))
        
    connectivity_count.sort(key=lambda x: x[1])
    with open("connectivity.txt", 'w+', encoding='utf-8') as f:
        f.write(str(connectivity_count))
    #plt.bar(x, y)
    #plt.show()

#=========Rendering graph===========

if rendering_graph:
    depo = Deposit(DATA_PATH)
    origin_i = int(filename.split('-')[0])
    origin_coord = depo.point_coord[origin_i]
    origin_str = "562"

    B = nx.Graph()
    weight_limit = 50
    slice_margin = 100

    # Reading the file
    max_weight = None
    pos = {origin_i: depo.point_coord[origin_i]}
    with open(readfile, 'r', encoding='utf-8') as f:
        while (True):
            ln1 = f.readline()[:-1]
            if not ln1:
                break
            ln2 = f.readline()
            ln2_eval = eval(ln2)

            # Assign the max edge weight, this is used for changing alpha
            if not max_weight:
                max_weight = ln2_eval[1]

            if (ln2_eval[1] >= weight_limit and
                depo.point_coord[ln2_eval[0]][2] > origin_coord[2] - slice_margin and 
                depo.point_coord[ln2_eval[0]][2] < origin_coord[2] + slice_margin):

                if (ln2_eval[0] == origin_i):
                    print("there is here")
                    B.add_node(ln1, color="red",size=2)#0.5*ln2_eval[1] / max_weight)
                else:
                    B.add_node(ln1, alpha=0.5)#0.5*ln2_eval[1] / max_weight)

                B.add_edge(origin_str, ln1, alpha=0.5)#0.5*ln2_eval[1] / max_weight)
                pos[ln1] = depo.point_coord[ln2_eval[0]][:-1]

    colors = range(20)
    options = {"pos": pos, 
        "with_labels": False, "font_size": 1,
        "node_color": "#A0CBE2", "node_size": 0.5,
        #"edge_color": colors, 
        "edge_cmap": plt.cm.Blues, "width": 0.2}

    nx.draw(B, **options)
    plt.show()
    #plt.savefig("graph.png", dpi=500)