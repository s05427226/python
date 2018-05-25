from numpy import *

def loadDataSet(filename):
    numFeat = len(open(filename).readline().split('\t')) - 1
    dataMat = []
    labelMat = []

    with open(filename) as f:
        for line in f:
            lineArr = []
            curLine = line.strip().split('\t')

            for i in range(numFeat):
                lineArr.append(float(curLine[i]))

            dataMat.append(lineArr)
            labelMat.append(float(curLine[-1])) 
    
    return dataMat,labelMat

def showDataMat(xMat,yMat,ws):
    import matplotlib.pyplot as plt

    fig= plt.figure()

    ax = fig.add_subplot(111)

    ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0],s=2,c='red')

    xCopy = xMat.copy()
    yHat = xCopy * ws

    ax.plot(xCopy[:,1],yHat)
    plt.show()


def standRegress(xArr,yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T

    xTx = xMat.T * xMat

    if linalg.det(xTx) == 0.0:
        print("the matrix is singular,cannot do inverse")
        return

    ws = xTx.I * (xMat.T * yMat)
    return ws


def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))

    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    
    xTx = xMat.T * (weights * xMat)

    if linalg.det(xTx) == 0.0:
        print("the matrix is singular,cannot do inverse")
        return
    
    ws = xTx.I * (xMat.T * (weights*yMat))

    return testPoint * ws

def lwlrTest(testArr,xArr,yArr,k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    
    return yHat



def Test():
    xArr,yArr = loadDataSet("ex0.txt")
    xMat = mat(xArr)
    yMat = mat(yArr)
    ws = standRegress(xMat,yMat)
    print(ws)

    # showDataMat(mat(dataMat),mat(labelMat),ws)

    yHat = xMat * ws
    print(corrcoef(yHat.T,yMat))


    yHat = lwlrTest(xArr,xMat,yMat,0.003)

    srtInd = xMat[:,1].argsort(0)
    xSort = xMat[srtInd][:,0,:]

    import matplotlib.pyplot as plt

    fig= plt.figure()

    ax = fig.add_subplot(111)

    ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0],s=2,c='red')

    ax.plot(xSort[:,1],yHat[srtInd])
    plt.show()

def rssError(yArr,yHatArr):
    return ((yArr - yHatArr) ** 2).sum()


def ridgeRegress(xMat,yMat,lam = 0.2):
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(denom) == 0.0:
        print("the matrix is singular,cannot do inverse")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws

def ridgeTest(xArr,yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean
    xMean = mean(xMat,0)
    xVar = var(xMat,0)
    xMat = (xMat - xMean) / xVar

    numTestPts = 30
    wMat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegress(xMat,yMat,exp(i-10))
        wMat[i,:] = ws.T
    return wMat

def stageWise(xArr,yArr,eps = 0.01,numIt = 100):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean
    xMat = regularize(xMat)

    m,n = shape(xMat)

    returnMat = zeros((numIt,n))

    ws = zeros((n,1))
    wsTest = ws.copy()
    wsMat = ws.copy()

    for i in range(numIt):
        print(ws.T)

    lowestError = inf;
    for j in range(n):
        for sign in [-1,1]:
            wsTest = ws.copy()
            wsTest[j] += eps * sign
            yTest = xMat * wsTest
            rssE = rssError(yMat.A,yTest.A)

            if rssE < lowestError:
                lowestError = rssE
                wsMax = wsTest
        
    ws = wsMax.copy

    returnMat[i,:] = ws.T
    return returnMat


def Test1():
    abX,abY = loadDataSet('abalone.txt')
    yHat01 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],0.1)
    yHat1 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],1)
    yHat10 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],10)

    print(rssError(abY[0:99],yHat01.T),rssError(abY[0:99],yHat1.T),rssError(abY[0:99],yHat10.T))

    yHat01 = lwlrTest(abX[100:199],abX[0:99],abY[0:99],0.1)
    yHat1 = lwlrTest(abX[100:199],abX[0:99],abY[0:99],1)
    yHat10 = lwlrTest(abX[100:199],abX[0:99],abY[0:99],10)

    print(rssError(abY[100:199],yHat01.T),rssError(abY[100:199],yHat1.T),rssError(abY[100:199],yHat10.T))

def Test2():
    abX,abY = loadDataSet('abalone.txt')
    rw = ridgeTest(abX,abY)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(rw)
    plt.show()


if __name__ == "__main__":
    pass
    # Test()
    # Test1()
    # Test2()