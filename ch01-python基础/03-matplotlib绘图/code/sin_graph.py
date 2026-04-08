import numpy as np
import matplotlib.pyplot as plt

# 准备数据
x = np.arange(0, 6, 0.1)
y = np.sin(x)

# 绘图
plt.plot(x, y)
plt.show()
