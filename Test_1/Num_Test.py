import pandas as pd
import math

def CreateDate():
    data = pd.read_csv('student.csv')
    #将缺失的数据用0代替，正则化，在原始数据也会生效
    data.replace(to_replace='NaN', value=0, regex=True, inplace=True)
    return data

#均值
def GetMean(data):
    Data_Len = len(data)
    print(Data_Len)
    sum = 0.
    for i in range(Data_Len):
        sum =sum + data.loc[i]
    data_mean = sum / Data_Len
    return data_mean

#方差
def GetVariance(data):
    Data_Len = len(data)
    Data_Mean = GetMean(data)
    sum = 0.
    for i in range(Data_Len):
        sum += (data.loc[i] - Data_Mean)**2
    return sum / Data_Len

#分位数
def GetQuantile(data, percent):
    if percent<0. or percent>1.:
        return None
    datalen_y = len(data)
    datalen_x = len(data.T)
    dataset = []
    # 向上去整
    c = math.ceil(datalen_y * percent)
    for i in range(datalen_x):
        datalist = []
        for j in range(datalen_y):
            datalist.append(data.iloc[j,i])
        datalist.sort()
        dataset.append(datalist[c])
    return dataset

if __name__ == '__main__':
    Pd_data = CreateDate()
    print("读取的数据为：\n",Pd_data)
    print('均值为：\n',GetMean(Pd_data))
    print("方差：\n",GetVariance(Pd_data))
    print("分位数：\n",GetQuantile(Pd_data, 0.5))

