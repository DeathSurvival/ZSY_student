import numpy as np
import matplotlib.pyplot as plt

#数据部分
x_data = [338., 333., 328., 207., 226., 25., 179., 60., 208., 606.]
y_data = [640., 633., 619., 393., 428., 27., 193., 66., 226., 1591.]

#背景图部分
x = np.arange(-200, -100, 1)
y = np.arange(-5, 5, 0.1)
z = np.zeros((len(x),len(y)))
X,Y = np.meshgrid(x,y)
for i in range(len(x)):
    for j in range(len(y)):
        b = x[i]
        w = y[j]
        z[j][i] = 0
        for n in range(len(x_data)):
            z[j][i] = z[j][i] + (y_data[n] - b - w*x_data[n])**2
        z[j][i] = z[j][i]/len(x_data)


#数据分析部分
b = -120
w = -4
lr = 1
itertion = 100000

b_history = [b]
w_history = [w]

lr_b = 0
lr_w = 0

for i in range(itertion):
    b_grad = 0.0
    w_grad = 0.0
    for n in range(len(x_data)):
        b_grad = b_grad - 2.0*(y_data[n] - b - w*x_data[n])*1.0
        w_grad = w_grad - 2.0*(y_data[n] - b - w*x_data[n])*x_data[n]

    lr_b = lr_b + b_grad ** 2
    lr_w = lr_w + w_grad ** 2

    b = b - lr/np.sqrt(lr_b) * b_grad
    w = w - lr/np.sqrt(lr_w) * w_grad

    b_history.append(b)
    w_history.append(w)

print(b_history[10000])
print(w_history[10000])

plt.contourf(x,y,z, 50, alpha=0.5, cmap=plt.get_cmap('jet'))    #制作背景图
plt.plot([-188.4], [2.67], 'x', ms=12, markeredgewidth=3,color='orange')    #标记终点
plt.plot(b_history, w_history, 'o-', ms=3, lw=1.5, color='black')   #制作点线
plt.xlim(-200, -100)
plt.ylim(-5, 5)
plt.xlabel(r'$b$',fontsize=16)
plt.ylabel(r'$w$',fontsize=16)
plt.show()