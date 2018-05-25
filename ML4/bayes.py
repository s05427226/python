def loadDataSet():
    postingList = [ ['my','dog','has','flea','problem','help','please'],\
                    ['maybe','not','take','him','to','dog','park','stupid'],\
                    ['my','dalmation','is','so','cute','i','love','him'],\
                    ['stop','posting','stupid','worthless','garbage'],\
                    ['mr','licks','ate','my','steak','how','to','stop','him'],\
                    ['quit','buying','worthless','dog','food','stupid'] ]
    
    classVec = [0,1,0,1,0,1]

    return postingList,classVec

def createVocabList(dataSet):
    vocabSet = set()
    for doc in dataSet:
        vocabSet = vocabSet | set(doc)

    return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):
    retVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            retVec[vocabList.index(word)] = 1
        else:
            print("the word:%s is not in my Vocabulary!" % word)
    return retVec

def bagOfWords2Vec(vocabList,inputSet):
    retVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            retVec[vocabList.index(word)] += 1
    return retVec

def trainNB0(trainMatrix,trainCateGory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])

    pAbusive = sum(trainCateGory)/float(numTrainDocs)

    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0

    for i in range(numTrainDocs):
        if trainCateGory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)

    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)

    if p1 > p0:
        return 1
    else:
        return 0
    
from numpy import *

def trainNB():
    postingList,classVec = loadDataSet()
    vocabList= createVocabList(postingList)
    tarinMat = []

    for postinDoc in postingList:
        tarinMat.append(setOfWords2Vec(vocabList,postinDoc))
    
    p0V,p1V,pAb = trainNB0(tarinMat,classVec)
    

    testEntry = ['love','my','dalmation']
    thisDoc = setOfWords2Vec(vocabList,testEntry)

    print(testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb))

    testEntry = ['stupid','garbage']
    thisDoc = setOfWords2Vec(vocabList,testEntry)
    print(testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb))

import re

def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*',bigString)
    return [ tok.lower() for tok in listOfTokens if len(tok) > 2 ]

def spamTest():
    docList = []
    classList = []
    fullText = []

    for i in range(1,23):
        wordList = textParse(open("./email/spam/%d.txt" % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        
        wordList = textParse(open("./email/ham/%d.txt" % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocabList = createVocabList(docList)
    trainingSet = list(range(44))
    testSet = []

    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    
    trainMat = []
    trainClasses = []

    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))

    errorCount = 0

    for docIndex in testSet:
        wordVec = setOfWords2Vec(vocabList,docList[docIndex])

        if classifyNB(array(wordVec),p0V,p1V,pSpam) != classList[docIndex]:
                errorCount += 1
        
    print("the error reate is:",float(errorCount)/len(testSet))


def trainNB2():
    
    mySent = "this book is the best book on python or M.L. i have laid eyes upon"
    regEx = re.compile('\\W*')
    listOfTokens = regEx.split(mySent)
    listOfTokens = [tok.lower() for tok in listOfTokens if len(tok) > 0]
    print(listOfTokens)

    print(regEx.split(open("./email/ham/6.txt").read()))

if __name__ == "__main__":
    # trainNB()
    # trainNB2()
    spamTest()