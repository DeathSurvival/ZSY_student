import pandas as pd
import numpy as np
import KNN

def CreateData():
    data = pd.read_csv('MissData.csv')
    return data

def getFillMissingByKNN(data):
    loc = {}    #使用字典记录空坐标，键代表纵向缺失位置，值代表横向缺失位置
    i = 0
    for columname in data.columns:
        if data[columname].count() != len(data):
            # 获取缺失数据的位置
            loc[i] = data[columname][data[columname].isnull().values == True].index.tolist()
        i = i + 1
    #此时的loc的值为{8: [1, 8]},表示在第9列的第2、9行分别有缺失
    colum_True = []     #用于储存纵向未缺失的位置
    colum_False = []    #用于储存纵向缺失的位置
    for colum_feature in range(len(data.columns)):
        if colum_feature in loc:
            colum_False.append(colum_feature)
        else:
            try:    #判断值是否为数字
                data.iloc[0, colum_feature].dtype
            except AttributeError:
                pass
            else:
                colum_True.append(colum_feature)
    i = 0
    for c in loc.values():
        index_True = []     #用于储存横向未缺失的位置
        index_False = []    #用于储存横向缺失的位置
        for index_feature in range(len(data.index)):
            if index_feature not in c:
                index_True.append(index_feature)
            else:
                index_False.append(index_feature)
        for t in index_False:
            dataLabel = data.iloc[index_True, colum_False[i]].values    #获得未缺失数据的数据类型
            dataInx = data.iloc[t, colum_True].T.values     #获得缺失数据的其他相关信息
            dataset = data.iloc[index_True, colum_True].values  #获得未缺失数据
            dataClass = KNN.classify0(dataInx, dataset, dataLabel, 3)   #调用KNN算法
            data.iloc[t, colum_False[i]] = dataClass
        i = i + 1
    return data

def GetFillMissing(data):
    print('填充前的数据：')
    print(data)
    print('使用均值填充后的数据：')
    data.fillna(data.mean(), inplace=True)  #使用均值进行填充
    print(data)
    print('使用KNN算法填充后的数据：')
    print(getFillMissingByKNN(data))

if __name__ == '__main__':
    data = CreateData()
    GetFillMissing(data)
