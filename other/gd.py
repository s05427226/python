from numpy import *

def loadDataSet(filename):
    datMat = []
    with open(filename) as f:
        for line in f.readlines():
            curline = line.strip().split('\t')
            datMat.append( [float(curline[0]), float(curline[1])] )
    return datMat


def gradientDescent(x,y):
    m,n = shape(x)
    ws = ones((n, 1))
    ita = 0.0001
    maxcycle = 1400
    for i in range(maxcycle):
        theta = x * ws
        ws = ws - ita * x.T * (y - theta)
    
    return ws

datMat = loadDataSet('testSet.txt')
datMat = mat(datMat)

ws = gradientDescent(datMat[:,0],datMat[:,1])

print(ws)

import matplotlib.pyplot  as plt

fig = plt.figure()

ax = fig.add_subplot(111)

ax.scatter(datMat[:, 0].tolist(),datMat[:, 1].tolist(), s = 5)

x = arange(-3.0,3.0,0.1)
j = 0
y = arange(-3.0,3.0,0.1)

for i in x:
    y[j] = i * ws[0]
    j += 1

ax.plot(x,y)

plt.show()

