from numpy import *

def loadDataSet():
    dataMat = []
    labelMat = []

    f = open('testSet.txt')
    for line in f.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))

    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn,classLabels):
    dataMat = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()

    m,n = shape(dataMat)
    alpha = 0.001
    maxCycle = 500
    weights = ones((n,1))

    for k in range(maxCycle):
        h = sigmoid(dataMat * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMat.transpose() * error

    return weights

def stocGradAscent0(dataMatrix,classLabel,numIter = 150):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)

    for j in range(numIter):
        
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0+j+i) + 0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabel[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])

    return weights

def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]

    xcord1 = []
    ycord1 = []

    xcord2 = []
    ycord2 = []

    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')

    x = arange(-3.0,3.0,0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x,y)

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def classifyVector(inX,weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:return 1.0
    else:return 0.0

def colicTest():
    fTrain = open('horseColicTraining.txt')
    fTest = open('horseColicTest.txt')
    trainingSet = []
    traingLabel = []

    for line in fTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        traingLabel.append(float(currLine[21]))

    trainWeight = stocGradAscent0(array(trainingSet),traingLabel,500)
    errorCount = 0
    numTestVec = 0.0

    for line in fTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr),trainWeight)) != int(currLine[21]):
            errorCount += 1

    errorRate = float(errorCount) / numTestVec
    print("the error rate of this test is :%f" % errorRate)
    return errorRate

def multiTest():
    numTest = 10
    errorSum = 0.0

    for k in range(numTest):
        errorSum += colicTest()
    
    print("after %d iteration the average error rate is :%f" %(numTest,errorSum/float(numTest)))

if __name__ == "__main__":
    
    dataMat,labelMat = loadDataSet()
    # plotBestFit(gradAscent(dataMat,labelMat).getA())
    # plotBestFit(stocGradAscent0(array(dataMat),labelMat))

    multiTest()