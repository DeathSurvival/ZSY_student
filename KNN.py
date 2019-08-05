from numpy import *
import operator

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group,labels

def classify0(inX, dataset, labels, k):
    dataSetSize = dataset.shape[0]  #放回矩阵行数
    '''
    tile([1.0, 2.0], (x, y)) 将列表[1.0, 2.0]变为矩阵
    每一行增加y-1个相同的元素，增加x-1行
    例如tile([1.0, 2.0], (2, 2))结果为
    [[1. 2. 1. 2.]
     [1. 2. 1. 2.]]
    '''
    diffMat = tile(inX, (dataSetSize, 1)) - dataset     #计算测试点与已知类别数据点的向量差
    sqDiffMat =diffMat ** 2     #向量差平方
    sqDistances = sqDiffMat.sum(axis = 1)   #没一行求和
    distances = sqDistances ** 0.5  #开根号
    sortedDisIndicies = distances.argsort()     #从小到大排序,以相对位置的结果返回结果
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDisIndicies[i]]   #得到此位置的标签
        #classCount.get(voteIlabel, 0)返回字典关键字voteIlabel对应的值,没有就放回0
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    #按关键字对应的数值从大到小排序
    sortedClassCount = sorted(classCount.items(),
                              key = operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

if(__name__ == '__main__'):
    gou, lab = createDataSet()
    print(classify0([1.0, 0.3], gou, lab, 2))