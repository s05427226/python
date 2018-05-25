from numpy import *
import operator
from os import listdir

def createDataSet():
    group = array([ [1.0,1.0],[1.0,1.1],[0,0],[0.0,0.1], ])
    labels = ['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1)) - dataSet

    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)

    distances = sqDistances ** 0.5
    distancesIndic = distances.argsort()

    classCount = {}

    for i in range(k):
        voteLabel = labels[distancesIndic[i]]
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1
    
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse = True)
    return sortedClassCount[0][0]

def file2Matrix(filename):
    f = open(filename)
    arrayOfLines = f.readlines()
    numberOfLines = len(arrayOfLines)
    retMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0

    for line in arrayOfLines:
        line = line.strip()
        listFromLine = line.split('\t')
        retMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    
    return retMat,classLabelVector

import matplotlib
import matplotlib.pyplot as plt

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(datingMat[:,1],datingMat[:,2],15*array(datingLabels),15.0*array(datingLabels))
# plt.show()

def autoNorm(datSet):
    minVals = datSet.min(0)
    maxVals = datSet.max(0)
    ranges = maxVals - minVals

    normDataSet = zeros(shape(datSet))

    m = datSet.shape[0]

    normDataSet = datSet - tile(minVals,(m,1))
    normDataSet = normDataSet / tile(ranges,(m,1))

    return normDataSet,ranges,minVals

def datingClassTest():
    hoRatio = 0.10
    datingMat,datingLabels = file2Matrix("datingTestSet2.txt")
    normDataSet,ranges,minVals = autoNorm(datingMat)
    m = normDataSet.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0

    for i in range(numTestVecs):
        classifierResult = classify0(normDataSet[i,:],normDataSet[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with:%d,the real answer is :%d" %(classifierResult,datingLabels[i]))
        if(classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is:%f" %(errorCount/float(numTestVecs)))

# datingClassTest()

def classifyPerson():
    resultList=['not at all','in small doses','in large doese']

    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))

    datingMap,datingLabels = file2Matrix("datingTestSet2.txt")
    normMat,ranges,minVals = autoNorm(datingMap)
    inArr = array([ffMiles,percentTats,iceCream])

    result = classify0((inArr - minVals)/ranges,normMat,datingLabels,3)

    print("you will problay like this person:",resultList[result-1])

def image2Vector(filename):
    returnVect = zeros((1,1024))
    f = open(filename)
    for i in range(32):
        lineStr = f.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    traingFileList = listdir('trainingDigits')
    m = len(traingFileList)

    trainMat = zeros((m,1024))

    for i in range(m):
        fileNameStr = traingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainMat[i,:] = image2Vector('trainingDigits/%s' % fileNameStr)

    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)

    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = image2Vector('testDigits/%s' %fileNameStr)

        result = classify0(vectorUnderTest,trainMat,hwLabels,5)

        print("the classifier came back with:%d,the real answer is:%d" %(result,classNumStr))
        if(result != classNumStr):errorCount += 1.0

    print("the total number of error is:%d" %errorCount)
    print("\nthe total error rate is:%f" % (errorCount/float(mTest)))

handwritingClassTest()
# classifyPerson()
# group,labels = createDataSet()
# print(classify0([0,0],group,labels,3))