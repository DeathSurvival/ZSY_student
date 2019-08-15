import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    '''
    :param fileName: 要导入的文件
    :return:矩阵，其中元素的数据类型也是矩阵
    '''
    dataMat = []
    file = open(fileName)
    for line in file.readlines():
        curLine = line.strip().split(' ')   #以空格符分割，输出为列表
        fltLine = []
        for i in curLine:   #将数值从字符形式变为浮点数形式
            i = float(i)
            fltLine.append(i)
        #fltLine = map(float,curLine)会出错
        dataMat.append(fltLine)
    return dataMat

def distEclud(vecA, vecB):
    '''
    :param vecA: 空间内的点A
    :param vecB: 空间内的点B
    :return: 点A和点B之间的距离
    '''
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))

def randCent(dataset, k):
    '''
    :param dataset: 数据集矩阵
    :param k: 质心个数
    :return: 结果矩阵，包含k个质心向量
    '''
    n = np.shape(dataset)[1]
    centroids = np.mat(np.zeros((k,n)))   #创建k行，n列的全0（浮点型）矩阵
    for j in range(n):
        minJ = min(dataset[:,j])    #第j列最小值
        rangeJ = float(max(dataset[:, j])- minJ)    #第j列最小值与最大值之差
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)  #在最大值与最小值之间随机取值
    return centroids

def KMeans(dataset, k, disMeas = distEclud, createCent = randCent):
    '''
    :param dataset: 数据集矩阵
    :param k: k个质心
    :param disMeas: 求距离算法
    :param createCent: 求随机点算法
    :return:数据集的质心矩阵，每个点属于哪一个簇及到此簇质心的距离
    '''
    m = np.shape(dataset)[0]
    #记录到哪一个质心跟进，并距离距离
    clusterAssment = np.mat(np.zeros((m, 2)))
    centroids = createCent(dataset, k)  #得到三个随机质心
    clusterChanged = True   #记录质心是否改变
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf    #np.inf表示正无穷
            minIndex = -1   #记录距离最小点对应的下标
            for j in range(k):
                distJI = disMeas(centroids[j,:],dataset[i,:])   #计算各个点与质心点的距离
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex,minDist**2  #记录到哪个质心更近，并记录距离
        #PLT(datamat, centroids)
        for cent in range(k):
            '''
                matrix矩阵名.A代表将 矩阵转化为array数组类型.
                np.nonzero()获得矩阵中不为0或者是True的下标, 第一列表示横，第二列表示纵
            '''
            ptsInClust = dataset[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent,:] = np.mean(ptsInClust, axis=0) #更新第cent簇数据的质心
    return centroids, clusterAssment

def biKmeans(dataset, k, disMeads=distEclud):
    m = np.shape(dataset)[0]
    clusterAssment = np.mat(np.zeros((m,2)))
    #计算所有数据点的均值,tolist()将数组或者矩阵转化为列表
    centroid0 = np.mean(dataset, axis=0).tolist()[0]
    centList = [centroid0]  #簇列表
    for j in range(m):
        clusterAssment[j, 1] = disMeads(np.mat(centroid0), dataset[j, :])**2 #mat()将列表装换为矩阵
    while (len(centList) < k):
        lowestSSE = np.inf  #定义无穷大
        for i in range(len(centList)):
            #ptsIncurrCluster是一个簇中所有元素的子数据集
            ptsIncurrCluster = dataset[np.nonzero(clusterAssment[:, 0].A == i)[0], :]
            #将子数据集划分为两个数据集得到的质心，和每个数据点属于的哪一个簇及到质心的距离
            centroidMat,splitClusAss = KMeans(ptsIncurrCluster, 2, disMeads)
            #sseSplit需要划分的字数据集在划分后的误差
            sseSplit = np.sum(splitClusAss[:, 1])
            # sseSplit未划分的子数据集的误差
            sseNotSplit = np.sum(clusterAssment[np.nonzero(clusterAssment[:, 0].A != i)[0], 1])
            if (sseSplit + sseNotSplit) < lowestSSE:    #选择最优的划分方式
                bestCentToSplit = i #记录最优划分是依据哪一个簇完成的
                bestNewCents = centroidMat  #记录最优的划分方式所划分的两个新的质心
                bestClustAss = splitClusAss.copy()  #记录最优划分方式所划分的子簇的信息
                lowestSSE = sseSplit + sseNotSplit  #重新定义最小误差
        #改变划分簇的质心编号
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        #更新簇质心，将原需要划分的簇质心去掉，将划分过后的两个新簇质心加入
        centList[bestCentToSplit] = bestNewCents[0, :][0].tolist()[0]
        centList.append(bestNewCents[1, :][0].tolist()[0])
        #更新簇
        clusterAssment[np.nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    return np.mat(centList), clusterAssment

def PLT(dataset, centroids):
    fig = plt.figure()
    plt.plot(dataset[:, 0], dataset[:, 1], '.', ms=12, markeredgewidth=3, color='blue')
    plt.plot(centroids[:, 0], centroids[:, 1], 'x', ms=12, markeredgewidth=3, color='orange')
    plt.show()

if __name__ == '__main__':
    datamat = np.mat(loadDataSet(fileName='testSet.txt'))
    centroids,clusterAssment = KMeans(datamat, 3)
    #PLT(datamat, centroids)
    print(centroids)
    centroid0, clusterAssment0 = biKmeans(datamat, 3)
    PLT(datamat, centroid0)
    print(centroid0)
