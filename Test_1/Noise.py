import pandas as pd
import numpy as np

def CreateData():
    data = pd.read_csv('test.csv')
    data.replace(to_replace='NaN', value=0, regex=True, inplace=True)
    dataset = data.values
    return dataset

def FilterNoiseThreshold(dataSet,maxThreshold,minThreshold):
    #利用最大最小的阈值来过滤噪声数据
    for i in range(len(dataSet)):
        for j in range(len(dataset[0])):
            if dataSet[i][j] < 65 or dataSet[i][j] > 85:
                dataSet[i][j] = np.nan
    return dataSet

def GetFrequencySpiltbox(data, box):
    '''
        将数据排序，并均分多个箱子，每个箱子box个数据
        将每个箱子的均值作为每个数据的数值
        也可用中位数或边界值，将箱子数据换成离他最近的边值
    '''
    data.sort()
    datalen = len(data)
    flag = datalen % int(box)   #用以判断数据是否被均分
    j = 0
    for i in range(int(datalen/box)):
        sum = 0.
        for c in range(int(box)):
            sum = sum + data[j + c] #计算一个箱子中数据的和
        databoxmean = sum/box   #获得均值
        for c in range(int(box)):   #修改列表中的数据
            data[j + c] = databoxmean
        j = j + box
    if flag:    #对末尾数据量不足的箱子进行处理
        sum = 0.
        for c in range(j, datalen):
            sum = sum + data[c]
        databoxmean = sum / (c - j + 1)
        for c in range(j, datalen):
            data[c] = databoxmean
    return data

def GetWidthSpiltbox(data,width):
    '''
        在整个属性值的区间上平均分布
        即每个箱的区间范围设定为一个常量，称为箱子的宽度
        例如：15,21,24,21,25,4,8,34,28的以10为宽度分箱的结果为：
        第一个箱子：4,8
        第二个箱子：15,21,21,24,25
        第三个箱子：28,34
    '''
    data.sort() #首先进行排序
    datastart = data[0] #储存箱子的一个边界
    sum = datastart #计算箱子中数据的和
    num = 1 #储存箱子中的数据量
    for i in range(1, len(data)):
        if (data[i] - datastart)>width: #超出了箱子的范围
            datamean = sum/num  #计算均值
            for j in range(i - num, i): #修改列表中数据
                data[j] = datamean
            datastart = data[i] #重新定义箱子边界
            num = 1
            sum = datastart
        else:   #数据仍在箱子中
            sum = sum + data[i]
            num = num + 1
    datamean = sum / num
    i = i + 1
    for j in range(i - num, i):
        data[j] = datamean
    return data

if __name__ == '__main__':
    dataset = CreateData()
    #print("过滤掉噪声后的数据为：\n",FilterNoiseThreshold(dataset, 85., 65.))
    #print('等频装箱均值填充过滤噪声数据:')
    print('等宽装箱均值填充过滤噪声数据:')
    for integer in dataset:
        #print(GetFrequencySpiltbox(integer, 4))
        print(GetWidthSpiltbox(integer, 5))
