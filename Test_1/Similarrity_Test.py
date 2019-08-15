import pandas as pd
import numpy as np

def CreateData():
    data_p = [1, 3, 2, 3, 4, 3, 3, 2]
    data_q = [1, 2, 4, 3, 2, 3, 2, 4]
    return data_p,data_q

def euclidean(data_p, data_q):
    '''
        在日常使用中，一般习惯于将相似度与1类比，相似度在数值上反映为0<=Similarity(X,y)<=1，越接近1，相似度越高；
        我们在使用欧几里得距离时，可以通过 1/（1+Distance(X,Y)）来贯彻上一理念
    '''
    sum = 0.
    for i in range(len(data_p)):
        sum = sum + (data_p[i] - data_q[i]) ** 2
    return 1 / (1 + sum ** 0.5)

def pearson(data_p, data_q):
    #皮尔逊相关度
    n = len(data_p)
    sumx,sumy,sumxsq,sumysq,sumxy = 0.,0.,0.,0.,0.
    for i in range(n):
        # 分别求出p，q的和
        sumx = sumx + data_p[i]
        sumy = sumy + data_q[i]
        # 分别求出p，q的平方和
        sumxsq = sumxsq + data_p[i]**2
        sumysq = sumysq + data_p[i]**2
        # 求出p，q的乘积和
        sumxy = sumxy + data_p[i] * data_q[i]
    #求出pearson相关系数
    up = sumxy - sumx*sumy/n
    down = ((sumxsq - pow(sumx,2)/n)*(sumysq - pow(sumy,2)/n))**.5
    #若down为零则不能计算，return 0
    if down == 0 :return 0
    r = up/down
    return r

def jaccard():
    #Jaccard系数
    a = [1, 0, 1, 1, 0, 1, 0, 0, 0]
    b = [0, 1, 0, 1, 1, 0, 0, 1, 1]
    c = 0
    for i in range(len(a)):
        if a[i] == 1 and a[i] == b[i]:
            c = c + 1
    return float(c)/len(a)

if __name__ == '__main__':
    data_p,data_q = CreateData()
    print('欧几里德距离：')
    print(euclidean(data_p,data_q))
    print('皮尔逊相关度：')
    print(pearson(data_p, data_q))
    print('Jaccard系数')
    print(jaccard())
