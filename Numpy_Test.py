import numpy as np

data = np.array([[2, 1, 3], [2, 8, 6]]) #创建一个2行三列的矩阵
data_1 = np.arange(0, 6).reshape(3,2)   #创建递增的(3,2)矩阵，只能从0开始,到5结束
data_2 = np.random.random((2,3))    #随机生成0-1数字矩阵

print(data)
print(data_1)
print(data_2)
print('返回矩阵的维数：',data.ndim,'数据类型为：',type(data.ndim))
print('返回矩阵的行列数：',data.shape,'数据类型为：',type(data.shape))
print('返回矩阵的大小（有多少个元素）：',data.size,'数据类型为：',type(data.size))

print('----------矩阵运算----------')
print(data)
print('data*10:',data * 10)
print("data**2:",data ** 2)
print("data>=3:",data >= 3)
print('矩阵相乘：', data.dot(data_1))
print('求矩阵元素和：',np.sum(data_1, axis=0)) #axis=0表示列
print('求矩阵元素最小值：',np.min(data_1,axis=1))    #axi=1表示行
print('求矩阵元素最大值：',np.max(data_1))
print("矩阵最小值位置：",np.argmin(data))
print("矩阵最大值位置：",np.argmax(data))
print('矩阵平均值：',np.mean(data))
print('矩阵中位数：',np.median(data))
print('矩阵逐个累加：',np.cumsum(data))
print('矩阵前后相减：',np.diff(data))
print('矩阵排序：',np.sort(data))    #逐行排序
print('矩阵转置：',np.transpose(data))   #或者data.T
print('矩阵选取：',np.clip(data, 3, 8))  #小于3的写成3，大于8的写成8

print('----------矩阵索引----------')
print(data)
print('第2行：',data[1,:])
print('第2列：',data[:,1])
print('第2行，第2,3列：',data[1,1:3])
for i in data:  #迭代每一行
    print(i)
for j in data.flat:
    print(j, end= ' ')

print()
print('----------矩阵合并----------')
print(data)
print(data_1)
print("上下合并后的矩阵：",np.vstack((data, data_1.T)))
print('左右合并后的矩阵：',np.hstack((data.T, data_1)))
data_A = np.arange(3)[np.newaxis, :]
data_B = np.arange(3)[:, np.newaxis]
print("将数列改为矩阵：", data_A)
print("将数列改为矩阵：", data_B)

print('----------矩阵分割----------')
data_F = np.arange(2, 26, 2).reshape((3, 4))
print(data_F)
print('分割：',np.split(data_F, 2, axis=1))    #均等分割
print('分割：',np.array_split(data_F, 3, axis=1))    #可以不均等分割
print("行向分割：",np.vsplit(data_F, 3))
print("纵向分割：",np.hsplit(data_F, 2))
data = data_F.copy()    #将data_F的内容复制给data
print(data)
