import numpy as np
import matplotlib.pyplot as plt


'''
x_data = [338., 333., 328., 207., 226., 25., 179., 60., 208., 606.]
y_data = [640., 633., 619., 393., 428., 27., 193., 66., 226., 1591.]
plt.scatter(x = x_data, y=y_data)
'''

x = np.linspace(-1, 1, 50)  #50个单位
y1 = 2*x + 1
y2 = x**2 + 1

plt.figure()    #用以在不同的图上展示
plt.plot(x, y1)

plt.figure(num=3, figsize=(8, 5)) #figsize（横，纵）
p1, = plt.plot(x, y2, label='up')
#color颜色，linewidt宽度，linestyle样式,label线的名字
p2, = plt.plot(x, y1, color='red', linewidth=2.0, linestyle=':', label='down')
plt.legend(handles=[p1, p2], labels=['blue', 'red'], loc='best')

plt.xlim((-2, 2))   #设置x坐标轴取值范围
plt.ylim((-1, 3))   #设置y坐标轴取值范围
plt.xlabel("X")
plt.ylabel('Y')

new_ticks = np.linspace(-1, 2, 5)
plt.xticks(new_ticks )  #更改下标
plt.yticks([-1, -0.4, 0, 1.3, 2.6],
           ["bad", "normal", r'$\alpha$', "good", '$very\ good$'])

#移动坐标轴
ax = plt.gca()
ax.spines['right'].set_color('none')    #右边坐标轴颜色为无色
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')   #使用下边坐标轴代替x轴
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data',0))    #将下边坐标轴设置在y轴0处
ax.spines['left'].set_position(('data',0))

plt.show()
