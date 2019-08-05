from math import log
import operator
import matplotlib.pyplot as plt

def createData():
    dataset = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    return dataset

def createDataSet():
    dataset = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ["特征1",'特征2']
    return dataset,labels

def getShang(dataset):
    num_data = len(dataset)     #获得队列的长度既有多少个数据
    labelCounts = {}    #定义一个空的集合
    for featVec in dataset:
        currentLabel = featVec[-1]  #读取队列对末尾元素
        if currentLabel not in labelCounts.keys():  #如果currentLabel不是集合的关键字
            labelCounts[currentLabel] = 0   #在集合中添加以currentLabel为关键字的元素，值为1
        labelCounts[currentLabel] += 1  #在集合中以currentLabel关键字对应的值加1
    shang = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/num_data
        shang -= prob * log(prob, 2)
    return shang

def splitDataSet(dataset, axis, value):     #axis表示特征值在列表的位置，value表示特征的值
    DataSet_Value = []
    DataSet_False = []
    for featVec in dataset:
        if featVec[axis] == value:
            reducendFeatVec = featVec[:axis]
            reducendFeatVec.extend(featVec[axis+1:])  #列表中元素分别单独的复制到新列表，而不是复制单独复制进去一个列表元素
            DataSet_Value.append(reducendFeatVec)   #添加一个列表元素
    return DataSet_Value

def ChooseBestDataSet(dataset):
    num_data = len(dataset[0]) - 1
    baseEntropy = getShang(dataset)
    bestInfogain = 0.0
    bestFeature = 0
    for i in range(num_data):
        featList = [example[i] for example in dataset] #获取dataSet的第i个所有特征
        uniqueVals = set(featList)  #使用featList创建集合（集合元素不会重复）
        newEntropy = 0.0
        for value in uniqueVals:
            subDataset = splitDataSet(dataset, i, value)
            prob = len(subDataset)/float(len(dataset))  #计算子集与原数据集的长度比
            newEntropy += prob * getShang(subDataset)
            infogain = newEntropy - baseEntropy #计算此次划分的信息增益
        if(infogain  > bestInfogain):
            bestInfogain = infogain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    '''
    依据关键值对应的值进行排序（从大到小），结果为列表，其中元素为元组，元组数目字典的数目
    每一个元组第一个元素是字典关键字，第二元素为字典关键字对应的值
    如果值相同，则相对位置也相同
    '''
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    print(sortedClassCount)
    return sortedClassCount[0][0]

def createTree(dataset, labels):
    classList = [example[-1] for example in dataset]    #获取所有类型编号
    if classList.count(classList[0]) == len(classList): #如果所有类型编号相同
        return classList[0] #返回类型编号
    if len(dataset[0]) == 1:    #如果特征值已经辨别完毕，但是还是有不同的类型
        return majorityCnt(classList)   #执行majorityCnt函数
    bestfeat = ChooseBestDataSet(dataset)   #获得最好分类特征的位置
    bestFeatLabel = labels[bestfeat]    #获得最好特征的名字标签
    myTree = {bestFeatLabel:{}}     #创建一个二重字典，其中一重元素的关键字为特征标签
    del(labels[bestfeat])   #删除标签类中特征对应的标签
    featValues = [example[bestfeat] for example in dataset]     #获取dataSet选定特征的所有可能值
    uniqueVals = set(featValues)    #将其变为集合，使其没有重合的元素
    for values in uniqueVals:
        subLabels = labels[:]   #复制标签
        myTree[bestFeatLabel][values] = createTree(splitDataSet(dataset, bestfeat, values),subLabels)   #递归
    return myTree

def getNumLeafs(myTree):
    numLeafs = 0
    fistStr = myTree.keys()[0]
    secondDict = myTree[fistStr]
    for key in secondDict.key():
        if type(secondDict[key]).__name__=='dict':
            numLeafs += getNumLeafs(secondDict)
        else:
            numLeafs += 1
    return numLeafs

decisionNode = dict(boxstyle='sawtooth',fc='0.8')
leafNode = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')

def getTreeDepath(mytree):
    maxDepth = 0
    firstStr = mytree.keys()[0]
    secondDict = mytree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth = 1 + getTreeDepath(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

def plotMidText(cntrpr, parentpt, txtString):
    xMid = (parentpt[0] - cntrpr[0])/2.0 + cntrpr[0]
    yMid = (parentpt[1] - cntrpr[1])/2.0 + cntrpr[1]
    createPlot.ax1.text(xMid, yMid, txtString)

def plotNode(nodeText, centerpt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeText, xy=parentPt, xycoords='axes fraction',
                            xytext=centerpt, textcoords='axes fraction',
                            va='center', ha='center', bbox=nodeType, arrowprops=arrow_args)

def plotTree(mytree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(mytree)
    depth = getTreeDepath(mytree)
    firstStr = mytree.keys()[0]
    cntrpt = (plotTree.xoff + (1.0 + float(numLeafs))/2.0/plotTree.totalW,plotTree.yoff)
    plotMidText(cntrpt, parentPt, nodeTxt)
    plotNode(firstStr, cntrpt, parentPt, decisionNode)
    secondDict = mytree[firstStr]
    plotTree.yoff = plotTree.yoff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrpt, str(key))
        else:
            plotTree.xoff = plotTree.xoff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xoff, plotTree.yoff), cntrpt, leafNode)
            plotMidText((plotTree.xoff, plotTree.yoff), cntrpt, str(key))
    plotTree.yoff = plotTree.yoff + 1.0/plotTree.totalD

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepath(inTree))
    plotTree.xoff = -0.5/plotTree.totalW
    plotTree.yoff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()

if __name__ == '__main__':
    dataset = createData()
    num = ChooseBestDataSet(dataset)
    print('依据第%d个特征值进行划分' % num)
    print('划分结果为：')
    featList = [example[num] for example in dataset] #获取dataSet的第i个所有特征
    uniqueValues = set(featList)
    for values in uniqueValues:
        print(splitDataSet(dataset, num, values))

    print(majorityCnt(['o', 'yes', 'yes', 'on']))

    newdataset,label = createDataSet()
    print(createTree(newdataset, label))
    #createPlot(createTree(newdataset, label))