# 1.3 Matplotlib 绘图

## 学习目标

- 学会用 Matplotlib 画出简单的折线图
- 学会添加标签、标题、图例
- 理解绘图在深度学习中的作用

## 核心概念

### Matplotlib 是什么？

Matplotlib 是 Python 的画图库。如果你想在屏幕上画一条曲线、一个柱状图，就用它。

为什么学习深度学习要学画图？两个原因：

1. **观察学习过程** - 训练神经网络时，我们会画"损失值随时间变化的曲线"，看看模型有没有在进步。
2. **理解数据** - 在动手之前，先看看数据长什么样（比如手写数字的图片）。

### 折线图

折线图是最常用的图：横坐标是 X，纵坐标是 Y，把每个 (X, Y) 点连成线。

## 原理解析

### 基本用法

```python
import matplotlib.pyplot as plt   # 给 pyplot 起个别名
import numpy as np

x = np.arange(0, 6, 0.1)   # 从 0 到 6，每隔 0.1 取一个数
y = np.sin(x)               # 对每个 x 求正弦值

plt.plot(x, y)              # 画图
plt.show()                  # 显示图片
```

- `plt.plot(x, y)` 告诉计算机："把 x 和 y 对应的点连起来"。
- `plt.show()` 才会真正把图显示在屏幕上。没有这一行，什么都看不到。

### 添加装饰

```python
plt.plot(x, y, label="sin")           # label 给线起个名字
plt.xlabel("x 轴")                     # 横坐标的标签
plt.ylabel("y 轴")                     # 纵坐标的标签
plt.title("正弦函数")                  # 图的标题
plt.legend()                           # 显示图例（就是线的名字）
plt.show()
```

### 画多条线

```python
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, label="sin")
plt.plot(x, y2, linestyle="--")  # linestyle 改变线的样式
plt.legend()
plt.show()
```

### 显示图片

Matplotlib 还能显示图片（比如手写数字的图片）：

```python
import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread("数字图片.png")    # 读取图片文件
plt.imshow(img)                 # 显示图片
plt.show()
```

## 代码实战

### 示例 1：画正弦曲线

```python
# sin_graph.py
import numpy as np
import matplotlib.pyplot as plt

# 准备数据
x = np.arange(0, 6, 0.1)   # 以 0.1 为单位，生成 0 到 6 的数据
y = np.sin(x)

# 绘图
plt.plot(x, y)
plt.show()
```

**代码解读：**
- `np.arange(0, 6, 0.1)` 生成 `[0, 0.1, 0.2, ..., 5.9]`，共 60 个点。
- `np.sin(x)` 对这 60 个点分别求正弦值。
- `plt.plot(x, y)` 把这 60 个点连成线。

### 示例 2：画正弦和余弦曲线

```python
# cos_graph.py
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 6, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, label="sin")
plt.plot(x, y2, linestyle="--", label="cos")   # 用虚线画 cos
plt.xlabel("x")
plt.ylabel("y")
plt.title("正弦和余弦")
plt.legend()
plt.show()
```

## 避坑指南

1. **忘记 `plt.show()`** - 这是新手最常犯的错误。`plt.plot()` 只是准备数据，`plt.show()` 才会真正把图画出来。
2. **中文显示问题** - 在某些系统上，中文标签可能显示为方块。如果遇到问题，可以暂时先用英文标签。
3. **数据点太少** - 如果曲线看起来是折线而不是光滑的，把步长改小（比如 `np.arange(0, 6, 0.01)`）。

## 课后思考

1. 修改代码，画出 `y = x²`（x 的平方）的图像，x 的范围是 -5 到 5。
2. 尝试改变线的颜色：在 `plt.plot()` 中添加 `color="red"` 参数。
3. 用 `plt.savefig("my_graph.png")` 把图片保存到文件。
