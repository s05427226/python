from math import log
import operator

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)

    labelCounts = {}

    for featureVec in dataSet:
        currentLabel = featureVec[-1]
        if currentLabel not in labelCounts:
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    ShannonEnt = 0
    
    for key in labelCounts:
        prob = labelCounts[key] / numEntries
        ShannonEnt -= prob * log(prob,2)
    
    return ShannonEnt

def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featureVec in dataSet:
        if featureVec[axis] == value:
            reduceFeatVec = featureVec[:axis]
            reduceFeatVec.extend(featureVec[axis+1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1

    for i in range(numFeatures):
        featList = [ example[i] for example in dataSet ]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i

    return bestFeature

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    
    sortedClassCount = sorted(classCount.items(),key=operator.attrgetter(1),reverse=True)

    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList = [ example[-1] for example in dataSet ]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeature = chooseBestFeatureToSplit(dataSet)
    bestFeatureLabel = labels[bestFeature]

    myTree = { bestFeatureLabel:{} }
    del(labels[bestFeature])
    featVal = [example[bestFeature] for example in dataSet ]
    uniqueVals = set(featVal)
    for val in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatureLabel][val] = createTree(splitDataSet(dataSet,bestFeature,val),subLabels)
    
    return myTree

def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]

    featIndex = featLabels.index(firstStr)

    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def createDataSet():
    dataSet = [ [1,1,'yes'],[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no'] ]
    labels = ['no surfacing','flippers']

    return dataSet,labels

def storeTree(inputTree,filename):
    import pickle
    f = open(filename,'wb')
    pickle.dump(inputTree,f)
    f.close()

def grabTree(filename):
    import pickle
    f = open(filename,"rb")
    return pickle.load(f)

if __name__ == "__main__":
    dataSet,labels=createDataSet()
    splitDataSet(dataSet,1,1)
    print(chooseBestFeatureToSplit(dataSet))
    print(createTree(dataSet,labels))

    # import treePlotter

    # # myTree = treePlotter.retrieveTree(0)
    # myTree = grabTree("1.txt")
    print(classify(myTree,labels,[1,1]))

    storeTree(myTree,"1.txt")

# print(calcShannonEnt(dataSet))