import numpy as np
from matplotlib import pylab as pl

# 定义训练数据
x = np.array([1,3,2,1,3])
y = np.array([14,24,18,17,27])

# 回归方程求取函数
def fit_1(x_data,y_data):
    if len(x) != len(y):
        return
    # 数据分析部分
    b = -120
    w = -4
    lr = 1
    itertion = 100000
    lr_b = 0
    lr_w = 0
    for i in range(itertion):
        b_grad = 0.0
        w_grad = 0.0
        for n in range(len(x)):
            b_grad = b_grad - 2.0 * (y_data[n] - b - w * x_data[n]) * 1.0
            w_grad = w_grad - 2.0 * (y_data[n] - b - w * x_data[n]) * x_data[n]
        lr_b = lr_b + b_grad ** 2
        lr_w = lr_w + w_grad ** 2
        b = b - lr / np.sqrt(lr_b) * b_grad
        w = w - lr / np.sqrt(lr_w) * w_grad
    return b,w

def fit_2(x,y):
    if len(x) != len(y):
        return
    numerator = 0.0
    denominator = 0.0
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    for i in range(len(x)):
        numerator += (x[i] - x_mean) * (y[i] - y_mean)
        denominator += np.square((x[i] - x_mean))
    print('numerator:', numerator, 'denominator:', denominator)
    b0 = numerator / denominator
    b1 = y_mean - b0 * x_mean
    return b0, b1

# 定义预测函数
def predit(x,b0,b1):
    return b0*x + b1

# 求取回归方程
b,w = fit_1(x,y)
print('Line is:y = %2.0fx + %2.0f'%(w,b))

# 预测
x_test = np.array([0.5,1.5,2.5,3,4])
y_test = np.zeros((1,len(x_test)))
for i in range(len(x_test)):
    y_test[0][i] = predit(x_test[i],w,b)

# 绘制图像
xx = np.linspace(0, 5)
yy = w*xx + b
pl.plot(xx,yy,'k-')
pl.scatter(x,y,cmap=pl.cm.Paired)
pl.scatter(x_test,y_test[0],cmap=pl.cm.Paired)
pl.show()