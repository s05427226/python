from numpy import *


myTree = {}


def loadDataSet(filename):
    dataMat = []
    with open(filename) as f:
        for line in f.readlines():
            curLine = line.strip().split('\t')
            fltLine = list(map(float, curLine))
            dataMat.append(fltLine)

    return dataMat


def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


def regLeaf(dataSet):
    return mean(dataSet[:, -1])


def regErr(dataSet):
    return var(dataSet[:, -1]) * shape(dataSet)[0]
    

def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    tolS = ops[0]
    tolN = ops[1]

    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)

    m, n = shape(dataSet)
    S = errType(dataSet)
    bestS = inf
    bestIndex = 0
    bestValue = 0

    for featIndex in range(n - 1):
        for splitVal in set((dataSet[:, featIndex].T.A.tolist())[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if shape(mat0)[0] < tolN or shape(mat1)[0] < tolN:
                continue

            newS = errType(mat0) + errType(mat1)

            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS

    if S - bestS < tolS:
        return None, leafType(dataSet)

    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if shape(mat0)[0] < tolN or shape(mat1)[0] < tolN:
        return None, leafType(dataSet)

    return bestIndex, bestValue


def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        return val

    retTree = {}

    retTree['spInd'] = feat
    retTree['spVal'] = val

    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)

    return retTree


def isTree(obj):
    return (type(obj).__name__ == 'dict')


def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return tree['right'] + tree['left']


def prune(tree, testData):
    if shape(testData)[0] == 0:
        return getMean(tree)
    if isTree(tree['right']) or isTree(tree['left']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'],tree['spVal'])
        errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) + \
        sum(power(rSet[:, -1] - tree['right'], 2))

        treeMean = (tree['left'] + tree['right'])/2.0

        errorMerge = sum(power(testData[:, -1] - treeMean, 2))

        if errorMerge < errorNoMerge:
            print('merging')
            return treeMean
        else:
            return tree
    else:
        return tree


def Test():
    myDat = loadDataSet('ex00.txt')
    myMat = mat(myDat)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(
        myMat[:, 0].flatten().A[0], myMat[:, 1].flatten().A[0], s=10, c='red')
    plt.show()

    myTree = reateTree(myMat)


def Test2():
    myDat = loadDataSet('ex0.txt')
    myMat = mat(myDat)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(
        myMat[:, 1].flatten().A[0], myMat[:, 2].flatten().A[0], s=10, c='red')
    plt.show()

    print(createTree(myMat))


def Test3():
    myDat = loadDataSet('ex2.txt')
    myMat = mat(myDat)

    myTree = createTree(myMat)

    print(myTree)

    myDatTest = loadDataSet('ex2test.txt')
    myMatTest = mat(myDatTest)

    print(prune(myTree, myMatTest))


def linerSolve(dataSet):
    m, n = shape(dataSet)

    X = mat(ones((m, n)))
    Y = mat(ones((m, 1)))

    X[:, 1:n] = dataSet[:, 0:n - 1]
    Y = dataSet[:, -1]

    xTx = X.T * X

    if linalg.det(xTx) == 0:
        raise NameEroor('this matrix is singlar,cannot do inverse.\n\
        try increaseing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y


def modelLeaf(dataSet):
    ws, X, Y = linerSolve(dataSet)
    return ws


def modelErr(dataSet):
    ws, X, Y = linerSolve(dataSet)
    yHat = X * ws

    return sum(power(Y - yHat, 2))


def Test4():
    myMat2 = loadDataSet('exp2.txt')
    myMat2 = mat(myMat2)

    myTree = createTree(myMat2, modelLeaf, modelErr, (1, 10))
    print(myTree)


def regTreeEval(model, inDat):
    return float(model)


def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1, n + 1)))
    X[:, 1 : n+1] = inDat
    return float(X * model)


def treeForeCast(tree, inDat, modelEval=regTreeEval):
    if not isTree(tree):
        return modelEval(tree, inDat)
    if inDat[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inDat, modelEval)
        else:
            return modelEval(tree['left'], inDat)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inDat, modelEval)
        else:
            return modelEval(tree['right'], inDat)


def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, mat(testData[i]), modelEval)

    return yHat


def Test5():
    trainMat = mat(loadDataSet('bikeSpeedVsIq_train.txt'))
    testMat = mat(loadDataSet('bikeSpeedVsIq_test.txt'))

    myTree = createTree(trainMat, ops=(1, 20))
    yHat = createForeCast(myTree, testMat[:, 0])
    print(corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])


    myTree = createTree(trainMat, modelLeaf, modelErr, (1, 20))
    yHat = createForeCast(myTree, testMat[:, 0],modelTreeEval)
    print(corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])

    ws, X, Y = linerSolve(trainMat)
    for i in range(shape(trainMat)[0]):
        yHat[i] = testMat[i, 0] * ws[1, 0] + ws[0, 0]
    print(corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])


from numpy import *
from tkinter import *

import matplotlib
matplotlib.use('TkAgg')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

root = Tk()

global reDraw
chkBtnVar = IntVar()
global tolNentry
global tolSentry

def reDraw(tolS, tolN):
    reDraw.f.clf()
    reDraw.a= reDraw.f.add_subplot(111)
    
    if chkBtnVar.get():
        if tolN < 2:
            tolN = 2
        myTree = createTree(reDraw.rawDat, modelLeaf, modelErr, (tolS, tolN))
        yHat = createForeCast(myTree, reDraw.testDat, modelTreeEval)
    else:
        myTree = createTree(reDraw.rawDat, ops=(tolS, tolN))
        yHat = createForeCast(myTree, reDraw.testDat)
    
    reDraw.a.scatter(reDraw.rawDat[:, 0].tolist(), reDraw.rawDat[:, 1].tolist(), s=5)
    reDraw.a.plot(reDraw.testDat, yHat, linewidth=2.0)
    reDraw.canvas.show()


def getInputs():
    try:
        tolN = int(tolNentry.get())
    except:
        tolN = 10
        print("enter interge for tolN")
        tolNentry.delete(0, END)
        tolNentry.insert(0, '10')
    
    try:
        tolS = float(tolSentry.get())
    except:
        tolS = 1.0
        print("enter float for tols")
        tolS.delete(0, END)
        tolS.insert(0, '1.0')
    
    return tolN, tolS


def drawNewTree():
    tolN,tolS = getInputs()
    reDraw(tolN,tolS)


Label(root, text='plot Place Holder').grid(row=0, columnspan=3)

Label(root, text='tolN').grid(row=1, column=0)

tolNentry = Entry(root)
tolNentry.grid(row=1, column=1)
tolNentry.insert(0, '10')

Label(root, text='tolS').grid(row=2, column=0)
tolSentry = Entry(root)
tolSentry.grid(row=2, column=1)
tolSentry.insert(0, '1.0')

Button(root, text='reDraw', command=drawNewTree).grid(row=1, column=2, rowspan=3)

chkBtn=Checkbutton(root, text='Model Tree', variable=chkBtnVar)
chkBtn.grid(row=3, column=0, columnspan=2)

reDraw.rawDat = mat(loadDataSet('sine.txt'))
reDraw.testDat = arange(min(reDraw.rawDat[:, 0]), max(reDraw.rawDat[:, 0]), 0.01 )

reDraw.f = Figure(figsize=(5, 4), dpi=100)
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
reDraw.canvas.show()
reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)

reDraw(1.0, 10)

root.mainloop()


# if __name__ == "__main__":
#     # Test()
#     # Test2()
#     # Test3()
#     # Test4()
#     # Test5()
#     treeExplore()
    