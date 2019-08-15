from math import log
import operator
import matplotlib.pyplot as plt

def createDataSet():
    dataset = [['teenager' ,'high', 'no' ,'same', 'no'],
   ['teenager', 'high', 'no', 'good', 'no'],
   ['middle_aged' ,'high', 'no', 'same', 'yes'],
   ['old_aged', 'middle', 'no' ,'same', 'yes'],
   ['old_aged', 'low', 'yes', 'same' ,'yes'],
   ['old_aged', 'low', 'yes', 'good', 'no'],
   ['middle_aged', 'low' ,'yes' ,'good', 'yes'],
   ['teenager' ,'middle' ,'no', 'same', 'no'],
   ['teenager', 'low' ,'yes' ,'same', 'yes'],
   ['old_aged' ,'middle', 'yes', 'same', 'yes'],
   ['teenager' ,'middle', 'yes', 'good', 'yes'],
   ['middle_aged' ,'middle', 'no', 'good', 'yes'],
   ['middle_aged', 'high', 'yes', 'same', 'yes'],
   ['old_aged', 'middle', 'no' ,'good' ,'no']]
    labels = ['age','input','student','level']
    return dataset,labels

#计算熵的公式
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

#分裂的子集合
def splitDataSet(dataset, axis, value):     #axis表示特征值在列表的位置，value表示特征的值
    DataSet_Value = []
    DataSet_False = []
    for featVec in dataset:
        if featVec[axis] == value:
            reducendFeatVec = featVec[:axis]
            reducendFeatVec.extend(featVec[axis+1:])  #列表中元素分别单独的复制到新列表，而不是复制单独复制进去一个列表元素
            DataSet_Value.append(reducendFeatVec)   #添加一个列表元素
    return DataSet_Value

#一次分裂
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

#不能完全分裂
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

#使用递归实现所有分裂
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

#计算决策树的叶子数
def getNumLeafs(myTree):
    numLeafs = 0    #叶子数
    sides = list(myTree.keys()) #节点信息
    firstStr = sides[0]
    secondDict = myTree[firstStr]   # 分支信息
    for key in secondDict.keys():  # 遍历所有分支
        if type(secondDict[key]).__name__ == 'dict':    #子树分支则递归计算
            numLeafs += getNumLeafs(secondDict[key])
        else:   # 叶子分支则叶子数+1
            numLeafs += 1
    return numLeafs

#计算决策树的深度
def getTreeDepth(myTree):
    maxDepth = 0    #最大深度
    sides = list(myTree.keys()) #节点信息
    firstStr = sides[0]
    secondDict = myTree[firstStr]   #分支信息
    for key in secondDict.keys():  #遍历所有分支
        if type(secondDict[key]).__name__ == 'dict':    #子树分支则递归计算
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   # 叶子分支则叶子数+1
            thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth   #更新最大深度
    return maxDepth

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

# ==================================================
# 输入：
#  nodeTxt:  终端节点显示内容
#  centerPt: 终端节点坐标
#  parentPt: 起始节点坐标
#  nodeType: 终端节点样式
# 输出：
#  在图形界面中显示输入参数指定样式的线段(终端带节点)
# ==================================================
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    '画线(末端带一个点)'
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)

# =================================================================
# 输入：
#  cntrPt:  终端节点坐标
#  parentPt: 起始节点坐标
#  txtString: 待显示文本内容
# 输出：
#  在图形界面指定位置(cntrPt和parentPt中间)显示文本内容(txtString)
# =================================================================
def plotMidText(cntrPt, parentPt, txtString):
    '在指定位置添加文本'
    # 中间位置坐标
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

# ===================================
# 输入：
#  myTree: 决策树
#  parentPt: 根节点坐标
#  nodeTxt: 根节点坐标信息
# 输出：
#  在图形界面绘制决策树
# ===================================
def plotTree(myTree, parentPt, nodeTxt):
    '绘制决策树'
    numLeafs = getNumLeafs(myTree)  #当前树的叶子数
    sides = list(myTree.keys()) #当前树的节点信息
    firstStr = sides[0]
    # 定位第一棵子树的位置(这是蛋疼的一部分)
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    # 绘制当前节点到子树节点(含子树节点)的信息
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    # 获取子树信息
    secondDict = myTree[firstStr]
    # 开始绘制子树，纵坐标-1。
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():  # 遍历所有分支
        # 子树分支则递归
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        # 叶子分支则直接绘制
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    # 子树绘制完毕，纵坐标+1。
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD

#在图形界面显示决策树inTree
def createPlot(inTree):
    # 创建新的图像并清空 - 无横纵坐标
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    # 树的总宽度 高度
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    # 当前绘制节点的坐标
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    # 绘制决策树
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()

if __name__ == '__main__':
    newdataset,label = createDataSet()
    Trees = createTree(newdataset, label)
    print(Trees)
    createPlot(Trees)