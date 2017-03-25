import numpy as np

"""
A note on data parsing:
Most of this is self-explanatory. Each basepair is a 4-bit byte consisting of a single 1 and three 0s.
What I want to point out is that for the negative test set, I am just taking every 17 basepairs as one
unit of data, rather than every possible 17-basepair stretch. I tried this at first; the datafile alone
is over 200 mb, and training the neural network was impossible. 
"""
def parse(l):
    Xt = []
    i = 0
    for item in l:
        Xt.append([])
        for ch in item:
            if ch == 'A':
                Xt[i].extend([0,0,0,1])
            elif ch == 'T':
                Xt[i].extend([0,0,1,0])
            elif ch == 'G':
                Xt[i].extend([0,1,0,0])
            elif ch == 'C':
                Xt[i].extend([1,0,0,0])
        i += 1
    X = np.array(Xt)
    return X


def get_data(filename):
    Xt =[]

    with open(filename) as f:
        i = 0
        for line in f:
            Xt.append([])
            for ch in line:
                if ch == 'A':
                    Xt[i].extend([0,0,0,1])
                elif ch == 'T':
                    Xt[i].extend([0,0,1,0])
                elif ch == 'G':
                    Xt[i].extend([0,1,0,0])
                elif ch == 'C':
                    Xt[i].extend([1,0,0,0])
            i += 1
    X = np.array(Xt)
    #print("Finished parsing positives")
    return X
"""
To be run only once. Makes a file that has every possible 17-length read of the negative data.
"""


def get_negatives():
    temp = [[]]

    i = 0
    with open("yeast-upstream-1k-negative.fa") as f:
        for line in f:
            if line[0:1] == '>':
                temp.append([])
                i +=1
            elif line[0:1] != '>':
                for ch in line:
                    if ch != '\n':
                        temp[i].extend(ch)

    X = []
    k = 0
    for i in temp:
        n = 0
        for j in i:
            X.append([])
            X[k].extend(i[n:n+17])
            n += 17
            k += 1
    new = [s for s in X if len(s) > 16]

    out = open('negative_formatted_less.txt','w')
    for item in new:
        out.write("%s\n"%item)


def read_negatives():
    X = []
    with open('negative_formatted_less.txt','r') as inf:
        for line in inf:
            X.append(eval(line))
    #print("Finished reading negatives")
    return X

"""
X1 = get_data()
l = read_negatives()
X = parse(l)
print("done parsing negatives")
print(X1)
print(X)
"""
