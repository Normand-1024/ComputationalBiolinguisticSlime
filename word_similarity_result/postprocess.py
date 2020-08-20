import os
import csv

limit = 30

for fn in os.listdir(os.getcwd()):
    if fn.endswith('.txt'):

        cnt = 30
        cosine = False
        if fn.endswith('cosine.txt'):
            cosine = True


        with open(fn, 'r') as fn2:
            #with open('postprocess/' + fn + '.csv', 'w+', newline='') as csf:
            with open('postprocess/' + fn + 'reformed.txt', 'w+') as writef:
                #csvwriter = csv.writer(csf)
                fn2.readline()
                #csvwriter.writerow(["weight", 'word'])

                while True:
                    ln = fn2.readline()

                    if not ln:
                        break

                    word = ln.split(':')[-1]
                    word = word.split('_')[0]

                    ln = fn2.readline()
                    #similarity = ln.split(':')[1]

                    for i in range(cnt):
                        writef.write(word + '\n')

                    cnt -= 1
                    ln = fn2.readline()