import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName, 'r')
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

def pca(dataMat, topNfeat=999999):
    meanVals = np.mean(dataMat, axis=0) #计算每一列的平均值
    meanRemoved = dataMat - meanVals    #去平均值
    conMat = np.cov(meanRemoved, rowvar=0)  #计算列与列之间的方差，得到协方差矩阵
    #计算矩阵np.mat(conMat)的特征值和特征向量，eigVals为特征值，eigVects为特征向量
    eigVals,eigVects = np.linalg.eig(np.mat(conMat))
    eigValInd = np.argsort(eigVals) #argsort函数返回的是数组值从小到大的索引值
    eigValInd = eigValInd[: -(topNfeat + 1): -1]    #-1表示负方向，取最大的topNfeat个索引
    redEigVects = eigVects[:, eigValInd]    #取索引值所对应的特征向量
    lowDataMat = meanRemoved * redEigVects  #使用特征向量与去均值后分向量的乘积结果为降维后的向量
    reconMat = (lowDataMat * redEigVects.T) + meanVals  #将原始数据转换到新空间
    return lowDataMat,reconMat

if __name__  == '__main__':
    dataMat = loadDataSet('testSet.txt')
    lowData,reconMat = pca(dataMat, 1)
    print('lowData:')
    print(lowData)
    print('reconMat:')
    print(reconMat)
    np.mat(dataMat)
    fig = plt.figure()
    plt.plot(np.mat(dataMat)[:, 0], np.mat(dataMat)[:, 1], '^', ms=12, markeredgewidth=3, color='red')
    #plt.plot(lowData[:, 0], lowData[:, 1], '.', ms=12, markeredgewidth=3, color='blue')
    plt.plot(reconMat[:, 0], reconMat[:, 1], 'x', ms=12, markeredgewidth=3, color='orange')
    plt.show()

