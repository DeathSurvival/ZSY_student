import numpy as np
import matplotlib.pyplot as plt

x_data = [338., 333., 328., 207., 226., 25., 179., 60., 208., 606.]
y_data = [640., 633., 619., 393., 428., 27., 193., 66., 226., 1591.]

x = np.linspace(0, 700, 50)
y = -188.4 + 2.67*x

sum = 0.
for i in range(10):
    sum += (y_data[i] + 188.4 - 2.67*x_data[i])**2
print(sum)

plt.scatter(x_data, y_data)
plt.plot(x,y)
plt.show()