import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def CreateData():
    dataset = [59, 84, 74, 69, 88, 83, 57, 66, 71, 93, 66, 84]
    return dataset

def Data_cut():
    #等宽法
    scores = [59, 84, 74, 69, 88, 83, 57, 66, 71, 93, 66, 84]
    df = pd.DataFrame({"Scores": scores})
    bins = [0, 60, 70, 80, 90, 100]
    group_names = ['very bad', 'bad', 'pass', 'good', 'very good']
    s = pd.cut(df["Scores"], bins, labels=group_names)
    df["labels"] = s
    print(df)

    # 用散点图表示，其中颜色按照codes分类
    plt.scatter(df.index, df["Scores"], cmap="Greens", c=pd.cut(scores, bins, labels=group_names).codes)
    plt.grid()  #显示网格
    plt.show()

def Data_qcut(data):
    #等频法
    df = pd.DataFrame({"data": data})
    cats = pd.qcut(df["data"], 3)   #将排序好的数据均分为三等分,每一个区间有4个数据
    df["qcut"] = cats
    print(df)
    print(df["qcut"].value_counts())    #记录每一区间的数量

    plt.scatter(df.index, df["data"], cmap="Reds", c=pd.qcut(data, 4).codes)
    plt.grid()
    plt.show()

def Data_Min_MAX(data):
    '''
        定义：也称为离差标准化，是对原始数据的线性变换，使得结果映射到0-1之间。
        本质：把数变为【0,1】之间的小数。
        转换函数：（X-Min）/(Max-Min)
        如果想要将数据映射到-1,1，则将公式换成：（X-Mean）/(Max-Min)
        一般用于在不涉及距离度量、协方差计算、数据不符合正太分布的时候
    '''
    data.sort()
    Min = data[0]
    Max = data[-1]
    dataset_new = []
    dif = (Max - Min) * 1.0
    for i in range(len(data)):
        dataset_new.append((data[i] - Min)/dif)
    return dataset_new

def Data_Z_score(data):
    '''
        定义：这种方法给与原始数据的均值（mean）和标准差（standard deviation）进行数据的标准化。经过处理的数据符合标准正态分布，即均值为0，标准差为1.
        本质：把有量纲表达式变成无量纲表达式。
        转换函数：（X-Mean）/(Standard deviation)
        其中，Mean为所有样本数据的均值。Standard deviation为所有样本数据的标准差。
        一般用于 分类、聚类算法中，需要使用距离来度量相似性的时候、或者使用PCA技术进行降维的时候
    '''
    sum = 0.
    data_len = len(data)
    for i in range(data_len):
        sum = sum + data[i]
    data_mean = sum / data_len
    sum = 0.
    for i in range(data_len):
        sum = sum + (data[i] - data_mean) ** 2
    sum = sum / data_len
    data_deviation = math.sqrt(sum)
    for i in range(data_len):
        data[i] = (data[i] - data_mean) / data_deviation
    return data

if __name__ == '__main__':
    dataset = CreateData()
    #Data_cut()
    #Data_qcut(dataset)
    print("0-1归一化后的数据为：")
    print(Data_Min_MAX(dataset))
    print('0均值标准化后的数据为：')
    print(Data_Z_score(dataset))