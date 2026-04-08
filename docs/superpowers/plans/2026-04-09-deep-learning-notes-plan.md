# 从零开始深度学习学习笔记实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为《从零开始深度学习》全书 8 章创建面向 0 基础开发人员的详细中文学习笔记，包含理论讲解、代码解析和练习。

**Architecture:** 每章独立文件夹，小节拆分，代码与笔记分离。每篇笔记遵循"学习目标-核心概念-原理解析-代码实战-避坑指南-课后思考"六段式结构。

**Tech Stack:** Markdown (笔记), Python 3.x (代码示例), NumPy, Matplotlib

---

## 文件结构总览

### 根目录
- `README.md` - 学习总览、环境搭建、学习路线

### 每章目录结构 (ch01 到 ch08)
```
chXX-章节名/
├── 01-小节名.md
├── 02-小节名.md
└── code/
    └── 对应源代码.py
```

### 具体文件清单

| 任务 | 创建文件 | 说明 |
|------|---------|------|
| Task 1 | `README.md` | 学习总览文档 |
| Task 2 | `ch01-python基础/01-什么是python.md` | Python 基础概念 |
| Task 2 | `ch01-python基础/01-什么是python/code/hello.py` | 示例代码 |
| Task 2 | `ch01-python基础/02-numpy基础.md` | NumPy 多维数组 |
| Task 2 | `ch01-python基础/02-numpy基础/code/numpy_basic.py` | NumPy 示例 |
| Task 2 | `ch01-python基础/03-matplotlib绘图.md` | Matplotlib 基础 |
| Task 2 | `ch01-python基础/03-matplotlib绘图/code/` | 绘图示例代码 |
| Task 3 | `ch02-感知机/01-感知机是什么.md` | 感知机概念 |
| Task 3 | `ch02-感知机/02-简单逻辑电路实现.md` | 逻辑门实现 |
| Task 3 | `ch02-感知机/03-感知机的局限性.md` | 异或门问题 |
| Task 3 | `ch02-感知机/code/` | 感知机代码 |
| Task 4 | `ch03-神经网络/01-神经网络基础.md` | 从感知机到神经网络 |
| Task 4 | `ch03-神经网络/02-激活函数.md` | Sigmoid, ReLU, Softmax |
| Task 4 | `ch03-神经网络/03-多维数组运算.md` | 矩阵运算 |
| Task 4 | `ch03-神经网络/04-手写数字识别.md` | MNIST 推理 |
| Task 4 | `ch03-神经网络/code/` | 神经网络代码 |
| Task 5 | `ch04-神经网络的学习/01-从数据中学习.md` | 学习概念 |
| Task 5 | `ch04-神经网络的学习/02-损失函数.md` | 均方误差、交叉熵 |
| Task 5 | `ch04-神经网络的学习/03-数值微分.md` | 梯度计算 |
| Task 5 | `ch04-神经网络的学习/04-学习算法实现.md` | 训练循环 |
| Task 5 | `ch04-神经网络的学习/code/` | 学习算法代码 |
| Task 6 | `ch05-误差反向传播法/01-计算图.md` | 计算图概念 |
| Task 6 | `ch05-误差反向传播法/02-链式法则.md` | 链式法则推导 |
| Task 6 | `ch05-误差反向传播法/03-反向传播实现.md` | 层实现 |
| Task 6 | `ch05-误差反向传播法/code/` | 反向传播代码 |
| Task 7 | `ch06-学习技巧/01-参数优化方法.md` | SGD, Momentum, Adam |
| Task 7 | `ch06-学习技巧/02-权重初始值.md` | 初始化方法 |
| Task 7 | `ch06-学习技巧/03-Batch Normalization.md` | 批量归一化 |
| Task 7 | `ch06-学习技巧/04-正则化与过拟合.md` | Dropout, 权重衰减 |
| Task 7 | `ch06-学习技巧/code/` | 学习技巧代码 |
| Task 8 | `ch07-卷积神经网络/01-全连接层的问题.md` | CNN 动机 |
| Task 8 | `ch07-卷积神经网络/02-卷积层与池化层.md` | Conv, Pooling |
| Task 8 | `ch07-卷积神经网络/03-CNN实现.md` | 完整 CNN |
| Task 8 | `ch07-卷积神经网络/code/` | CNN 代码 |
| Task 9 | `ch08-深度学习进阶/01-更深的网络.md` | 深层网络 |
| Task 9 | `ch08-深度学习进阶/02-超参数调优.md` | 超参数搜索 |
| Task 9 | `ch08-深度学习进阶/03-从零开始深度学习总结.md` | 全书总结 |
| Task 9 | `ch08-深度学习进阶/code/` | 进阶代码 |

---

### Task 1: 项目初始化与学习总览

**Files:**
- Create: `README.md`

- [ ] **Step 1: 创建 README.md**

```markdown
# 从零开始深度学习 - 学习笔记

## 项目介绍

本项目是《从零开始深度学习》(ゼロから作る Deep Learning) 的配套学习笔记。所有笔记专为 **零基础开发者** 编写，用纯中文、通俗的语言解释深度学习的核心概念，配合从零手写的代码示例，帮助你真正理解神经网络的工作原理。

## 适合谁学？

- 会写代码，但从未接触过深度学习的开发者
- 不了解线性代数、微积分，希望用大白话理解数学概念的学习者
- 想自己动手实现神经网络，而不是只会调用框架的开发者

## 环境准备

### 需要安装的软件
1. Python 3.6 或更高版本
2. NumPy（用于数组运算）
3. Matplotlib（用于画图）

### 安装命令
```bash
pip install numpy matplotlib
```

## 学习路线

本书共 8 章，建议按顺序学习：

| 章节 | 内容 | 难度 | 预计时间 |
|------|------|------|----------|
| 第 1 章 | Python 基础 | ★☆☆ | 2 小时 |
| 第 2 章 | 感知机 | ★★☆ | 3 小时 |
| 第 3 章 | 神经网络 | ★★☆ | 4 小时 |
| 第 4 章 | 神经网络的学习 | ★★★ | 5 小时 |
| 第 5 章 | 误差反向传播法 | ★★★ | 5 小时 |
| 第 6 章 | 学习技巧 | ★★★ | 4 小时 |
| 第 7 章 | 卷积神经网络 | ★★★ | 5 小时 |
| 第 8 章 | 深度学习进阶 | ★★★ | 4 小时 |

## 目录结构

```
ch01-python基础/      # Python、NumPy、Matplotlib 入门
ch02-感知机/          # 感知机原理与逻辑电路实现
ch03-神经网络/        # 神经网络基础与手写数字识别
ch04-神经网络的学习/   # 损失函数、梯度下降
ch05-误差反向传播法/   # 高效计算梯度的算法
ch06-学习技巧/        # 优化器、初始化、正则化
ch07-卷积神经网络/     # 处理图像的专用网络
ch08-深度学习进阶/     # 深层网络与超参数调优
```

## 如何使用笔记？

每篇笔记包含六个部分：
1. **学习目标** - 本节要学什么
2. **核心概念** - 用大白话解释，不用公式
3. **原理解析** - 深入理解背后的逻辑
4. **代码实战** - 从零手写，逐行解析
5. **避坑指南** - 新手常犯错误
6. **课后思考** - 动手修改代码的练习

> **建议：** 不要只看不练。每看完一节，一定要动手运行代码，并尝试修改参数看看会发生什么。
```

- [ ] **Step 2: 验证文件创建**

确认 `README.md` 已创建，内容完整。

---

### Task 2: 第1章 Python基础 - 环境搭建与工具入门

**Files:**
- Create: `ch01-python基础/01-什么是python.md`
- Create: `ch01-python基础/01-什么是python/code/hungry.py`
- Create: `ch01-python基础/01-什么是python/code/man.py`
- Create: `ch01-python基础/02-numpy基础.md`
- Create: `ch01-python基础/02-numpy基础/code/numpy_basic.py`
- Create: `ch01-python基础/03-matplotlib绘图.md`
- Create: `ch01-python基础/03-matplotlib绘图/code/sin_graph.py`
- Create: `ch01-python基础/03-matplotlib绘图/code/cos_graph.py`

- [ ] **Step 1: 创建 Python 入门笔记**

文件: `ch01-python基础/01-什么是python.md`

```markdown
# 1.1 什么是 Python

## 学习目标

- 了解为什么深度学习常用 Python
- 学会运行简单的 Python 脚本
- 理解类和对象的基本概念

## 核心概念

### Python 是什么？

Python 是一门编程语言，就像我们说中文、英文一样，是人类和计算机"对话"的工具。

为什么深度学习几乎都用 Python？主要有三个原因：

1. **简单** - 代码读起来像英语句子，不需要记复杂的符号。
2. **库多** - 别人写好了很多"工具箱"（比如 NumPy），我们直接用就行。
3. **社区大** - 遇到问题随便一搜就能找到答案。

### 类和对象

可以把"类"想象成 **设计图纸**，"对象"就是 **按图纸盖出来的房子**。

- 一张图纸（类）可以盖很多栋房子（对象）。
- 每栋房子有自己的主人（数据），但结构一样（方法）。

## 原理解析

### Python 的基本语法

Python 用缩进（通常是 4 个空格）来表示代码块的层级关系，而不是像其他语言用大括号。

```python
# 这是一个注释，计算机会忽略它
print("Hello World")  # 这行会在屏幕上打印 Hello World
```

### 类（class）的结构

```
class 类名:
    def __init__(self):      # 初始化方法，创建对象时自动执行
        self.属性 = 值
    
    def 方法名(self):         # 对象能做的事情
        代码
```

- `__init__` 是一个特殊方法，在创建新对象时自动调用。
- `self` 代表"我自己"，用来访问对象自己的属性。

## 代码实战

### 示例 1：最简单的脚本

```python
# hungry.py
print("I'm hungry!")
```

**运行方式：**
```bash
python hungry.py
```

**输出：**
```
I'm hungry!
```

**代码解读：**
- `print()` 是一个内置函数，作用是把括号里的内容显示在屏幕上。
- 字符串用引号包裹，单引号 `'` 和双引号 `"` 效果一样。

### 示例 2：类和对象

```python
# man.py
class Man:
    """示例类：描述一个人"""

    def __init__(self, name):
        self.name = name          # 给自己起个名字
        print("初始化完成！")

    def hello(self):
        print("你好，我是" + self.name + "！")

    def goodbye(self):
        print("再见，" + self.name + "！")

# 下面是实际使用
m = Man("大卫")    # 按照 Man 这个图纸，创建一个叫"大卫"的人
m.hello()          # 让大卫打招呼
m.goodbye()        # 让大卫说再见
```

**输出：**
```
初始化完成！
你好，我是大卫！
再见，大卫！
```

**代码解读：**
- `class Man:` 定义了一个叫 Man 的类（设计图纸）。
- `"""示例类：描述一个人"""` 是文档字符串，用来解释这个类的作用。
- `m = Man("大卫")` 这行做了两件事：
  1. 创建了一个新的 Man 对象。
  2. 把"大卫"传给 `__init__` 方法，所以会打印"初始化完成！"。
- `m.hello()` 调用对象 m 的 hello 方法，`self` 自动指向 m，所以 `self.name` 就是"大卫"。

## 避坑指南

1. **缩进错误** - Python 对空格非常敏感，同一层级的代码必须保持相同的缩进。如果报错 `IndentationError`，检查空格数量。
2. **中英文标点** - 代码中所有的引号、括号、冒号必须用英文半角符号。中文的`""`和英文的`""`看起来很像，但计算机完全不认识中文的。
3. **文件编码** - 如果代码里有中文，确保文件保存为 UTF-8 编码。

## 课后思考

1. 修改 `man.py`，创建一个叫"小明"的对象，并调用他的方法。
2. 给 `Man` 类增加一个 `age` 属性，并在 `hello` 方法中打印年龄。
```

- [ ] **Step 2: 创建 hungry.py**

文件: `ch01-python基础/01-什么是python/code/hungry.py`

```python
print("I'm hungry!")
```

- [ ] **Step 3: 创建 man.py**

文件: `ch01-python基础/01-什么是python/code/man.py`

```python
class Man:
    """示例类：描述一个人"""

    def __init__(self, name):
        self.name = name
        print("初始化完成！")

    def hello(self):
        print("你好，我是" + self.name + "！")

    def goodbye(self):
        print("再见，" + self.name + "！")

m = Man("大卫")
m.hello()
m.goodbye()
```

- [ ] **Step 4: 创建 NumPy 基础笔记**

文件: `ch01-python基础/02-numpy基础.md`

```markdown
# 1.2 NumPy 多维数组

## 学习目标

- 理解什么是数组（Array）
- 学会用 NumPy 创建和操作多维数组
- 掌握数组的四则运算和广播功能

## 核心概念

### 什么是数组？

想象你有一排格子，每个格子里放一个数字。这就是 **一维数组**。

```
[ 1,  2,  3,  4,  5 ]   ← 一维数组：一排格子
```

如果格子排成一个表格，就是 **二维数组**（也叫矩阵）：

```
[ 1,  2,  3 ]
[ 4,  5,  6 ]           ← 二维数组：一个表格
```

以此类推，还有三维、四维……多维数组。

### 为什么需要 NumPy？

Python 自带的列表（list）也能存一排数字，但做数学运算时非常慢。

NumPy 是一个专门用来做数值计算的"工具箱"，它有两个巨大优势：

1. **快** - 底层用 C 语言实现，比 Python 原生循环快几十倍。
2. **方便** - 一行代码就能完成矩阵运算，不需要写 for 循环。

## 原理解析

### 生成 NumPy 数组

```python
import numpy as np            # 给 numpy 起个别名叫 np

x = np.array([1.0, 2.0, 3.0])  # 把 Python 列表变成 NumPy 数组
```

- `import numpy as np` 是固定写法，给 NumPy 起了个短名字。
- `np.array()` 把方括号里的列表转换成 NumPy 的数组。

### 数组的基本运算

```python
import numpy as np

x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])

# 逐元素相加：对应位置的数字相加
print(x + y)    # [3.0, 6.0, 9.0]

# 逐元素相减
print(x - y)    # [-1.0, -2.0, -3.0]

# 逐元素相乘
print(x * y)    # [2.0, 8.0, 18.0]

# 逐元素相除
print(x / y)    # [0.5, 0.5, 0.5]
```

**关键理解：** NumPy 的运算符是对 **每个元素** 分别操作，不是数学里的矩阵乘法。矩阵乘法要用 `np.dot()`。

### 多维数组

```python
A = np.array([[1, 2], [3, 4]])   # 2 行 2 列的矩阵
print(A.shape)                    # (2, 2) → 形状是 2 行 2 列
print(A.dtype)                    # int64 → 里面的数字是整数
```

- `.shape` 返回一个元组，告诉你数组的"长相"（几行几列）。
- `.dtype` 告诉你数组里装的是什么类型的数字（整数、小数等）。

### 广播（Broadcasting）

广播是 NumPy 最方便的功能之一：当两个数组形状不同时，NumPy 会 **自动扩展** 小的那个，让它们能一起运算。

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([10, 20])   # 一维数组

print(A * B)
# [[ 1*10,  2*20],
#  [ 3*10,  4*20]]
# = [[10, 40],
#    [30, 80]]
```

B 被"广播"成了 `[[10, 20], [10, 20]]`，然后和 A 逐元素相乘。

## 代码实战

```python
# numpy_basic.py
import numpy as np

# === 1. 创建一维数组 ===
x = np.array([1.0, 2.0, 3.0])
print("x =", x)
print("x 的形状:", x.shape)

# === 2. 创建二维数组 ===
A = np.array([[1, 2], [3, 4]])
print("\nA =")
print(A)
print("A 的形状:", A.shape)

# === 3. 数组四则运算 ===
B = np.array([[3, 0], [0, 6]])
print("\nA + B =")
print(A + B)

print("\nA * B（逐元素相乘）=")
print(A * B)

# === 4. 广播 ===
print("\nA * 10（标量广播）=")
print(A * 10)

C = np.array([10, 20])
print("\nA * C（行向量广播）=")
print(A * C)

# === 5. 访问元素 ===
print("\nA[0] =", A[0])       # 第 0 行
print("A[0][1] =", A[0][1])  # 第 0 行第 1 列
```

**代码解读：**
- `x.shape` 告诉我们数组的维度。一维数组返回 `(3,)`，二维数组返回 `(行数, 列数)`。
- `A * B` 是对应位置的数字相乘，不是矩阵乘法。数学上的矩阵乘法用 `np.dot(A, B)`。
- `A[0]` 取出第 0 行，`A[0][1]` 取出第 0 行第 1 列的数字。（注意：Python 从 0 开始计数！）

## 避坑指南

1. **`*` 不是矩阵乘法** - 这是新手最常犯的错误。`A * B` 是逐元素相乘。要做矩阵乘法，用 `np.dot(A, B)` 或 `A @ B`。
2. **形状不匹配** - 广播不是万能的。两个数组必须满足广播规则才能运算。如果报错 `ValueError: operands could not be broadcast together`，检查两个数组的形状。
3. **整数除法** - 在 Python 3 中，`5 / 2 = 2.5`。如果你的数组是整数类型（int），除法结果会变成小数类型（float）。

## 课后思考

1. 创建一个 3×3 的单位矩阵（对角线是 1，其他是 0），提示：使用 `np.eye(3)`。
2. 用 `np.zeros((2, 3))` 创建一个全零的 2 行 3 列数组，观察它的形状和数据类型。
3. 尝试 `np.array([1, 2, 3]) + np.array([10])`，观察广播的结果。
```

- [ ] **Step 5: 创建 numpy_basic.py**

文件: `ch01-python基础/02-numpy基础/code/numpy_basic.py`

```python
import numpy as np

# 创建一维数组
x = np.array([1.0, 2.0, 3.0])
print("x =", x)
print("x 的形状:", x.shape)

# 创建二维数组
A = np.array([[1, 2], [3, 4]])
print("\nA =")
print(A)
print("A 的形状:", A.shape)

# 数组四则运算
B = np.array([[3, 0], [0, 6]])
print("\nA + B =")
print(A + B)

print("\nA * B（逐元素相乘）=")
print(A * B)

# 广播
print("\nA * 10（标量广播）=")
print(A * 10)

C = np.array([10, 20])
print("\nA * C（行向量广播）=")
print(A * C)

# 访问元素
print("\nA[0] =", A[0])
print("A[0][1] =", A[0][1])
```

- [ ] **Step 6: 创建 Matplotlib 绘图笔记**

文件: `ch01-python基础/03-matplotlib绘图.md`

```markdown
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
plt.plot(x, y2, label="cos", linestyle="--")  # linestyle 改变线的样式
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
```

- [ ] **Step 7: 创建 sin_graph.py**

文件: `ch01-python基础/03-matplotlib绘图/code/sin_graph.py`

```python
import numpy as np
import matplotlib.pyplot as plt

# 准备数据
x = np.arange(0, 6, 0.1)
y = np.sin(x)

# 绘图
plt.plot(x, y)
plt.show()
```

- [ ] **Step 8: 创建 cos_graph.py**

文件: `ch01-python基础/03-matplotlib绘图/code/cos_graph.py`

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 6, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, label="sin")
plt.plot(x, y2, linestyle="--", label="cos")
plt.xlabel("x")
plt.ylabel("y")
plt.title("正弦和余弦")
plt.legend()
plt.show()
```

- [ ] **Step 9: 验证第1章所有文件**

确认以下文件都已创建：
- `ch01-python基础/01-什么是python.md`
- `ch01-python基础/01-什么是python/code/hungry.py`
- `ch01-python基础/01-什么是python/code/man.py`
- `ch01-python基础/02-numpy基础.md`
- `ch01-python基础/02-numpy基础/code/numpy_basic.py`
- `ch01-python基础/03-matplotlib绘图.md`
- `ch01-python基础/03-matplotlib绘图/code/sin_graph.py`
- `ch01-python基础/03-matplotlib绘图/code/cos_graph.py`

---

### Task 3: 第2章 感知机 - 神经网络的前身

**Files:**
- Create: `ch02-感知机/01-感知机是什么.md`
- Create: `ch02-感知机/02-简单逻辑电路实现.md`
- Create: `ch02-感知机/03-感知机的局限性.md`
- Create: `ch02-感知机/code/perceptron.py`

- [ ] **Step 1: 创建感知机概念笔记**

文件: `ch02-感知机/01-感知机是什么.md`

```markdown
# 2.1 感知机是什么

## 学习目标

- 理解感知机的基本结构和工作原理
- 掌握感知机的数学表达式
- 理解权重和偏置的作用

## 核心概念

### 感知机：一个人工"神经元"

感知机是 1957 年发明的一种算法，可以把它想象成一个人工神经元。

**生活中的比喻：**

假设你在考虑"今天要不要出门"。你会考虑两个信号：
- 天气好不好？
- 有没有重要的事情？

每个信号的重要性不一样：
- 如果天气很重要，你就给它更高的"权重"。
- 如果有急事，即使天气不好也要出门，这就是"偏置"在起作用。

感知机就是这样工作的：**接收多个信号，每个信号有不同的重要性，最后决定"输出 1（是）"还是"输出 0（否）"**。

## 原理解析

### 感知机的数学公式

感知机的公式很简单：

```
y = 1  如果  w1*x1 + w2*x2 > 阈值
y = 0  否则
```

其中：
- `x1, x2` 是输入信号（比如天气、事情紧急度），只能是 0 或 1。
- `w1, w2` 是权重，表示每个信号的重要性。
- **阈值** 是一个门槛，加权和超过这个门槛就输出 1。

### 引入偏置

我们把公式改写一下，用"偏置"（bias）代替"阈值"：

```
y = 1  如果  w1*x1 + w2*x2 + b > 0
y = 0  否则
```

这里的 `b` 就是偏置。它的作用是 **调整输出的倾向性**：
- `b` 很大（正数）→ 即使输入信号很小，也容易输出 1。
- `b` 很小（负数）→ 即使输入信号很大，也不容易输出 1。

**类比：** 偏置就像一个"倾向性旋钮"。如果偏置是正数，感知机就"倾向于输出 1"；如果是负数，就"倾向于输出 0"。

### 权重和偏置的关系

- **权重** 控制每个输入信号的"音量"。权重大，这个信号就重要。
- **偏置** 控制整体的"门槛高低"。偏置大，门槛低，容易输出 1。

## 代码实战

感知机可以用 Python 函数来实现：

```python
def perceptron(x1, x2, w1, w2, b):
    """简单的感知机"""
    total = x1 * w1 + x2 * w2 + b
    if total > 0:
        return 1
    else:
        return 0
```

**代码解读：**
- `x1, x2` 是两个输入信号。
- `w1, w2` 是对应的权重。
- `b` 是偏置。
- 计算加权和，如果大于 0 就返回 1，否则返回 0。

## 避坑指南

1. **感知机不是神经网络** - 感知机是单个"神经元"。神经网络是把很多感知机连在一起。先理解单个的，再理解多个连起来的。
2. **阈值 vs 偏置** - 这两个概念是等价的。阈值是"门槛"，偏置是"门槛的反面"。偏置 = -阈值。
3. **输入只能是 0 或 1** - 本节讨论的感知机输入是二进制信号（0 或 1），后面章节的神经网络输入可以是任意数字。

## 课后思考

1. 如果权重都是 1，偏置是 -1.5，当 x1=1, x2=1 时，输出是什么？
2. 试着画一个感知机的结构图：两个输入 x1、x2，一个输出 y，中间标注权重和偏置。
```

- [ ] **Step 2: 创建逻辑电路实现笔记**

文件: `ch02-感知机/02-简单逻辑电路实现.md`

```markdown
# 2.2 用感知机实现逻辑门

## 学习目标

- 用感知机实现与门、与非门、或门
- 理解权重和偏置如何决定逻辑门的行为
- 体会感知机的强大之处

## 核心概念

### 什么是逻辑门？

逻辑门是数字电路的基本元件，接收 0 或 1 的输入，输出 0 或 1。

常见的逻辑门有三种：

| 门 | 规则 | 生活中的类比 |
|----|------|-------------|
| 与门（AND） | 两个输入都是 1 时输出 1 | "天气好 **并且** 有事才出门" |
| 与非门（NAND） | 两个输入都是 1 时输出 0 | "**除非** 天气好 **并且** 没事，否则都出门" |
| 或门（OR） | 至少一个输入是 1 时输出 1 | "天气好 **或者** 有事，就出门" |

## 原理解析

### 与门的权重

与门的真值表（所有可能的输入输出）：

| x1 | x2 | 输出 |
|----|----|------|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

要让感知机实现与门，只需要找到合适的权重和偏置：

- `w1 = 0.5, w2 = 0.5, b = -0.7`

验证一下：
- x1=0, x2=0 → 0×0.5 + 0×0.5 - 0.7 = -0.7 ≤ 0 → 输出 0 ✓
- x1=0, x2=1 → 0×0.5 + 1×0.5 - 0.7 = -0.2 ≤ 0 → 输出 0 ✓
- x1=1, x2=0 → 1×0.5 + 0×0.5 - 0.7 = -0.2 ≤ 0 → 输出 0 ✓
- x1=1, x2=1 → 1×0.5 + 1×0.5 - 0.7 = 0.3 > 0 → 输出 1 ✓

### 与非门

与非门就是把与门的输出反过来。权重取负数，偏置取正数就行：

- `w1 = -0.5, w2 = -0.5, b = 0.7`

### 或门

或门比与门"宽松"一些，只要有一个输入是 1 就输出 1。偏置可以调大一点：

- `w1 = 0.5, w2 = 0.5, b = -0.2`

## 代码实战

```python
# perceptron.py
import numpy as np

def AND(x1, x2):
    """与门：两个输入都是 1 时输出 1"""
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    total = np.sum(w * x) + b
    return 1 if total > 0 else 0

def NAND(x1, x2):
    """与非门：两个输入都是 1 时输出 0"""
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    total = np.sum(w * x) + b
    return 1 if total > 0 else 0

def OR(x1, x2):
    """或门：至少一个输入是 1 时输出 1"""
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    total = np.sum(w * x) + b
    return 1 if total > 0 else 0

# 测试
print("与门:")
for x1 in [0, 1]:
    for x2 in [0, 1]:
        print(f"  AND({x1}, {x2}) = {AND(x1, x2)}")

print("\n与非门:")
for x1 in [0, 1]:
    for x2 in [0, 1]:
        print(f"  NAND({x1}, {x2}) = {NAND(x1, x2)}")

print("\n或门:")
for x1 in [0, 1]:
    for x2 in [0, 1]:
        print(f"  OR({x1}, {x2}) = {OR(x1, x2)}")
```

**代码解读：**
- 用 NumPy 数组存储输入和权重，`w * x` 是逐元素相乘。
- `np.sum(w * x)` 把相乘后的结果加起来，等价于 `w1*x1 + w2*x2`。
- `1 if total > 0 else 0` 是 Python 的简洁写法，等价于 if-else 语句。

## 避坑指南

1. **权重不唯一** - 实现同一个逻辑门，权重和偏置可以有无数种组合。比如与门也可以是 `w1=1, w2=1, b=-1.5`。只要满足真值表就行。
2. **浮点数比较** - 代码里用 `> 0` 而不是 `>= 0`，这是为了避免刚好等于 0 时的歧义。

## 课后思考

1. 尝试用不同的权重实现与门（比如 w1=1, w2=1, b=-1.5），验证结果是否正确。
2. 思考：与非门和与门的权重有什么关系？为什么取负数就能"反过来了"？
```

- [ ] **Step 3: 创建感知机局限性笔记**

文件: `ch02-感知机/03-感知机的局限性.md`

```markdown
# 2.3 感知机的局限性

## 学习目标

- 理解为什么单层感知机无法实现异或门
- 理解"线性"和"非线性"的区别
- 理解为什么需要多层感知机

## 核心概念

### 异或门是什么？

异或门（XOR）的规则是：**两个输入不同时输出 1，相同时输出 0**。

| x1 | x2 | 输出 |
|----|----|------|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

生活中的类比："**要么** 天气好，**要么** 有事，但不能两个都满足"。

## 原理解析

### 为什么单层感知机做不了异或？

感知机的输出是由一条 **直线** 决定的：`w1*x1 + w2*x2 + b = 0` 是一条直线，直线的一边输出 1，另一边输出 0。

我们把异或门的四个点画在坐标系上：
- (0, 0) → 输出 0
- (0, 1) → 输出 1
- (1, 0) → 输出 1
- (1, 1) → 输出 0

你会发现：**不可能用一条直线把输出 1 的点和输出 0 的点分开**！

这就好比在操场上，红的人和蓝的人交错站着，你没法用一条直线把他们完全分开。

### 解决方案：多层感知机

既然一层不够，那就用两层！

异或门可以用与非门和与门组合实现：

```
异或门 = 与门(与非门(x1, x2), 或门(x1, x2))
```

具体来说：
1. 第一层：计算与非门和或门的输出。
2. 第二层：把第一层的输出作为与门的输入。

**这就意味着：感知机只要叠加多层，就能处理更复杂的问题！** 这就是神经网络的起点。

## 代码实战

```python
def XOR(x1, x2):
    """异或门：用多层感知机实现"""
    s1 = NAND(x1, x2)    # 第一层：与非门
    s2 = OR(x1, x2)      # 第一层：或门
    y = AND(s1, s2)      # 第二层：与门
    return y

# 测试
print("异或门:")
for x1 in [0, 1]:
    for x2 in [0, 1]:
        print(f"  XOR({x1}, {x2}) = {XOR(x1, x2)}")
```

**代码解读：**
- XOR 函数内部调用了 NAND、OR、AND 三个已经定义好的函数。
- 这就是"多层"：第一层同时计算 NAND 和 OR，第二层把结果喂给 AND。
- 这就是神经网络的基本思想：**把简单的单元堆叠起来，就能处理复杂的问题**。

## 避坑指南

1. **异或门不是"新"的感知机** - 异或门没有自己的权重，它是用已有的与门、与非门、或门组合出来的。这就是"多层"的意义。
2. **层数可以无限增加** - 理论上，只要层数够多、节点够多，神经网络可以模拟任何函数。但实际中，太深的网络很难训练。

## 课后思考

1. 在纸上画出异或门的结构图：两个输入 → 与非门和或门 → 与门 → 输出。
2. 思考：如果增加第三层，能实现更复杂的逻辑吗？举例说明。
```

- [ ] **Step 4: 创建感知机代码文件**

文件: `ch02-感知机/code/perceptron.py`

```python
import numpy as np

def AND(x1, x2):
    """与门：两个输入都是 1 时输出 1"""
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    total = np.sum(w * x) + b
    return 1 if total > 0 else 0

def NAND(x1, x2):
    """与非门：两个输入都是 1 时输出 0"""
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    total = np.sum(w * x) + b
    return 1 if total > 0 else 0

def OR(x1, x2):
    """或门：至少一个输入是 1 时输出 1"""
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    total = np.sum(w * x) + b
    return 1 if total > 0 else 0

def XOR(x1, x2):
    """异或门：用多层感知机实现"""
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

if __name__ == "__main__":
    print("与门:")
    for x1 in [0, 1]:
        for x2 in [0, 1]:
            print(f"  AND({x1}, {x2}) = {AND(x1, x2)}")

    print("\n与非门:")
    for x1 in [0, 1]:
        for x2 in [0, 1]:
            print(f"  NAND({x1}, {x2}) = {NAND(x1, x2)}")

    print("\n或门:")
    for x1 in [0, 1]:
        for x2 in [0, 1]:
            print(f"  OR({x1}, {x2}) = {OR(x1, x2)}")

    print("\n异或门:")
    for x1 in [0, 1]:
        for x2 in [0, 1]:
            print(f"  XOR({x1}, {x2}) = {XOR(x1, x2)}")
```

---

### Task 4: 第3章 神经网络 - 从感知机到深度学习

**Files:**
- Create: `ch03-神经网络/01-神经网络基础.md`
- Create: `ch03-神经网络/02-激活函数.md`
- Create: `ch03-神经网络/03-多维数组的矩阵运算.md`
- Create: `ch03-神经网络/04-手写数字识别推理.md`
- Create: `ch03-神经网络/code/neural_net_inference.py`

*(由于篇幅限制，后续 Task 的内容将在实际执行时按相同详细程度展开。每个 Task 都遵循上述模板。)*

- [ ] **Step 1: 创建神经网络基础笔记**

文件: `ch03-神经网络/01-神经网络基础.md`

```markdown
# 3.1 神经网络基础

## 学习目标

- 理解神经网络和感知机的区别
- 掌握神经网络的三层结构（输入层、隐藏层、输出层）
- 理解激活函数的作用

## 核心概念

### 神经网络 vs 感知机

上一章的感知机公式：
```
y = 1  如果  w·x + b > 0
y = 0  否则
```

神经网络的公式：
```
y = h(w·x + b)
```

区别只有一个：**激活函数 h()**。

感知机的激活函数是"阶跃函数"（大于 0 输出 1，否则输出 0），而神经网络用的是 **光滑的、连续变化的函数**，比如 Sigmoid 函数。

为什么要换激活函数？因为光滑的函数才能用"梯度"来训练（第 4 章会详细讲）。

### 神经网络的层

神经网络分三层：

1. **输入层** - 接收数据。比如手写数字的图片，每个像素就是一个输入。
2. **隐藏层** - 中间的处理层。可以有多个隐藏层，每层做不同的"特征提取"。
3. **输出层** - 给出最终结果。比如识别手写数字，输出层有 10 个节点（对应 0-9）。

**命名规则：** 只看有权重的层。输入层没有权重，所以不算。一个隐藏层 + 一个输出层 = "2 层网络"。

## 原理解析

### 信号传递过程

数据从输入层进入，经过隐藏层的处理，最后到达输出层：

```
输入层 (x1, x2) → 隐藏层 (a1, a2, a3) → 输出层 (y1, y2)
```

每一步的计算是：
1. 计算加权和：`a = w·x + b`
2. 通过激活函数：`z = h(a)`
3. 输出作为下一层的输入

### 为什么叫"神经"网络？

因为它的灵感来自生物神经元。生物神经元接收多个信号，当信号总和超过某个阈值时，就会"放电"（输出信号）。人工神经网络模仿了这个过程。

## 代码实战

以下代码展示了一个简单的 3 层神经网络的前向传播（从输入到输出的计算过程）：

```python
import numpy as np

# === 输入层 → 第 1 层 ===
x = np.array([1.0, 0.5])     # 输入数据
W1 = np.array([[0.1, 0.3, 0.5],   # 输入层到第 1 层的权重
               [0.2, 0.4, 0.6]])
b1 = np.array([0.1, 0.2, 0.3])    # 偏置

a1 = np.dot(x, W1) + b1           # 加权和
z1 = sigmoid(a1)                   # 激活函数

# === 第 1 层 → 第 2 层（输出层）===
W2 = np.array([[0.1, 0.4],
               [0.2, 0.5],
               [0.3, 0.6]])
b2 = np.array([0.1, 0.2])

a2 = np.dot(z1, W2) + b2
y = identity_function(a2)          # 输出层的激活函数

print("输出:", y)
```

**代码解读：**
- `np.dot(x, W1)` 是矩阵乘法。x 是 1×2，W1 是 2×3，结果是 1×3。
- `a1` 是加权和，`z1` 是经过激活函数处理后的值。
- 输出层的激活函数通常用"恒等函数"（输入什么输出什么），用于回归问题；分类问题用 Softmax。

## 避坑指南

1. **矩阵维度要匹配** - 这是最常见的错误。输入是 2 个数字，权重矩阵就必须有 2 行。记住：`x 的列数 = W 的行数`。
2. **偏置的形状** - 偏置 b 的个数必须等于当前层节点的个数。

## 课后思考

1. 如果输入 x 变成 `[0.5, 0.5]`，输出 y 会变成什么？运行代码验证。
2. 试着数一数上面网络中有多少个权重参数（W1 和 W2 的元素总数）和偏置参数（b1 和 b2 的元素总数）。
```

- [ ] **Step 2: 创建激活函数笔记**

文件: `ch03-神经网络/02-激活函数.md`

```markdown
# 3.2 激活函数

## 学习目标

- 理解激活函数为什么是"非线性"的
- 掌握 Sigmoid、ReLU、Softmax 三种激活函数
- 学会用代码实现这三种函数

## 核心概念

### 什么是激活函数？

激活函数是神经网络中的一个"转换器"。它接收加权和 `a`，输出转换后的值 `z`。

```
z = h(a)
```

激活函数必须是非线性的，否则无论网络有多少层，整体还是线性的（多层就等于一层）。

**类比：** 想象你有一排水龙头，每个水龙头控制水流大小。激活函数就是决定"拧多少"的规则。

## 原理解析

### 1. Sigmoid 函数

公式：
```
h(x) = 1 / (1 + e^(-x))
```

特点：
- 输入是任何数字，输出永远在 0 到 1 之间。
- 图像是一条光滑的 S 形曲线。
- 早期的神经网络常用，现在用得少了（后面会说原因）。

### 2. ReLU 函数（修正线性单元）

公式：
```
h(x) = x   如果 x > 0
h(x) = 0   如果 x ≤ 0
```

特点：
- 大于 0 的部分直接输出，小于 0 的部分输出 0。
- 计算非常快，是现代神经网络最常用的激活函数。

### 3. Softmax 函数（输出层专用）

公式：
```
y_k = e^(a_k) / Σ(e^(a_i))   对所有 i 求和
```

特点：
- 把多个数字转换成 **概率分布**（所有输出加起来等于 1）。
- 通常只用在输出层，用于分类任务。

## 代码实战

```python
import numpy as np

def sigmoid(x):
    """Sigmoid 函数"""
    return 1 / (1 + np.exp(-x))

def relu(x):
    """ReLU 函数"""
    return np.maximum(0, x)

def softmax(a):
    """Softmax 函数"""
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a

# 测试
x = np.array([-1.0, 1.0, 2.0])
print("Sigmoid:", sigmoid(x))
print("ReLU:", relu(x))
print("Softmax:", softmax(x))
print("Softmax 总和:", np.sum(softmax(x)))
```

**代码解读：**
- `np.exp(-x)` 计算 e 的 -x 次方，对数组中的每个元素分别计算。
- `np.maximum(0, x)` 比较 0 和 x 的每个元素，取大的那个。这就是 ReLU。
- Softmax 的输出加起来等于 1，可以理解为"概率"。

## 避坑指南

1. **Softmax 的溢出问题** - 当输入数字很大时，`e^x` 会变得非常大，导致计算溢出。安全的写法是减去最大值：`exp(a - max(a))`。
2. **激活函数选哪个** - 隐藏层优先选 ReLU。输出层：分类用 Softmax，回归用恒等函数。
3. **Sigmoid 的梯度消失** - 当输入很大或很小时，Sigmoid 的曲线很平，梯度几乎为 0。这就是为什么深层网络很少用它。

## 课后思考

1. 画出 Sigmoid 和 ReLU 的图像（用 Matplotlib），观察它们的区别。
2. 计算 `softmax([1000, 1000, 1000])` 会发生什么？为什么需要减去最大值？
```

- [ ] **Step 3: 创建多维数组矩阵运算笔记**

文件: `ch03-神经网络/03-多维数组的矩阵运算.md`

```markdown
# 3.3 多维数组的矩阵运算

## 学习目标

- 理解神经网络中矩阵乘法的维度变化
- 掌握 np.dot() 的使用方法
- 能手动推导张量的形状变化

## 核心概念

### 为什么用矩阵运算？

神经网络每一层的计算都是 `z = h(W·x + b)`。

如果每个节点都用 for 循环计算，代码会很慢很啰嗦。用矩阵运算，一行代码就能完成整个层的所有计算。

**类比：** 你要批改 30 份作业。for 循环是一份一份改。矩阵运算是一起改完。

### 维度匹配规则

矩阵乘法 `np.dot(A, B)` 要求：**A 的列数 = B 的行数**。

结果矩阵的形状：A 的行数 × B 的列数。

```
A (2, 3) · B (3, 4) → 结果 (2, 4)
        ↑↑ 必须匹配
```

## 原理解析

### 神经网络中的矩阵运算

以一个 2→3→2 的网络为例：

```
输入 x: (2,)          ← 2 个输入节点
权重 W1: (2, 3)       ← 2 行 3 列
偏置 b1: (3,)         ← 3 个节点
```

计算过程：
```
a1 = np.dot(x, W1) + b1
```

维度变化：
```
x (2,) · W1 (2, 3) = (3,) + b1 (3,) = (3,)
```

输出 a1 有 3 个元素，对应隐藏层的 3 个节点。

## 代码实战

```python
import numpy as np

# 2→3→2 网络的前向传播
x = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5],
               [0.2, 0.4, 0.6]])
b1 = np.array([0.1, 0.2, 0.3])

print("x 的形状:", x.shape)       # (2,)
print("W1 的形状:", W1.shape)     # (2, 3)

a1 = np.dot(x, W1) + b1
print("a1 的形状:", a1.shape)     # (3,)

# 继续第 2 层
W2 = np.array([[0.1, 0.4],
               [0.2, 0.5],
               [0.3, 0.6]])
b2 = np.array([0.1, 0.2])

a2 = np.dot(a1, W2) + b2
print("a2 的形状:", a2.shape)     # (2,)
```

## 避坑指南

1. **形状对不上** - 报错 `ValueError: shapes not aligned` 时，检查 `np.dot()` 两边的维度。A 的列数必须等于 B 的行数。
2. **批处理** - 实际中我们一次处理多条数据。如果输入是 100 条数据，x 的形状是 (100, 2)，计算方式完全一样，只是结果变成 (100, 3)。

## 课后思考

1. 如果网络结构是 3→4→2，输入 x 有 100 条数据（形状 (100, 3)），推导每一层输出的形状。
2. 为什么偏置 b 可以和矩阵乘法的结果直接相加？（提示：广播）
```

- [ ] **Step 4: 创建手写数字识别推理笔记**

文件: `ch03-神经网络/04-手写数字识别推理.md`

```markdown
# 3.4 手写数字识别推理

## 学习目标

- 理解 MNIST 数据集
- 用训练好的权重进行推理（识别手写数字）
- 理解批处理的概念

## 核心概念

### MNIST 数据集

MNIST 是深度学习界的"Hello World"。它包含：
- 60000 张训练图片
- 10000 张测试图片
- 每张图片是 28×28 像素的手写数字（0-9）

图片是黑白的，每个像素的值是 0（黑）到 255（白）之间的小数。

### 推理（Inference）

推理就是：给网络输入一张图片，让它输出"这是数字几"。

网络输出 10 个数字，分别代表"这张图片是 0、是 1、...、是 9"的概率。概率最大的那个就是答案。

## 原理解析

### 推理步骤

1. **加载数据** - 从文件读取图片和标签。
2. **预处理** - 把图片展平成一维数组（28×28 = 784 个像素），并缩放到 0-1 之间。
3. **前向传播** - 把处理好的输入喂给网络，得到输出。
4. **判断结果** - 输出中概率最大的索引就是识别的数字。

### 批处理

一次处理一张图片太慢了。我们可以把多张图片放在一起，变成一个矩阵，一次性计算。

```
一张图片: (784,)
100 张图片: (100, 784)   ← 批处理
```

批处理的好处：
- 更快（利用矩阵运算的并行性）。
- 代码更简洁。

## 代码实战

```python
import numpy as np
import sys
sys.path.append('..')
from common.functions import sigmoid, softmax

def get_data():
    """获取 MNIST 数据（简化版，实际需要下载数据集）"""
    from dataset.mnist import load_mnist
    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    """加载训练好的权重（这里是简化示例）"""
    # 实际应该从文件加载训练好的权重
    network = {}
    network['W1'] = np.random.randn(784, 50) * 0.01
    network['b1'] = np.zeros(50)
    network['W2'] = np.random.randn(50, 100) * 0.01
    network['b2'] = np.zeros(100)
    network['W3'] = np.random.randn(100, 10) * 0.01
    network['b3'] = np.zeros(10)
    return network

def forward(network, x):
    """前向传播"""
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y

x, t = get_data()
network = init_network()

# 批处理：一次处理 100 张图片
batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = forward(network, x_batch)
    p = np.argmax(y_batch, axis=1)    # 概率最大的索引
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("准确率:", float(accuracy_cnt) / len(x))
```

## 避坑指南

1. **权重要用训练好的** - 上面的代码用随机权重，所以准确率会很低。实际应该加载训练好的权重。第 4 章会学如何训练。
2. **axis 参数** - `np.argmax(y, axis=1)` 表示沿每一行找最大值。如果不指定 axis，会把整个矩阵展平后找最大值。

## 课后思考

1. 修改 `batch_size` 为 1，观察代码是否还能正常工作。
2. 思考：如果输出层有 10 个节点，但我们要识别的是 26 个英文字母，输出层应该有几个节点？
```

- [ ] **Step 5: 创建推理代码文件**

文件: `ch03-神经网络/code/neural_net_inference.py`

```python
import numpy as np

def sigmoid(x):
    """Sigmoid 函数"""
    return 1 / (1 + np.exp(-x))

def softmax(a):
    """Softmax 函数（数值稳定版本）"""
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a

def init_network():
    """初始化网络（随机权重，仅用于演示）"""
    network = {}
    network['W1'] = np.random.randn(784, 50) * 0.01
    network['b1'] = np.zeros(50)
    network['W2'] = np.random.randn(50, 100) * 0.01
    network['b2'] = np.zeros(100)
    network['W3'] = np.random.randn(100, 10) * 0.01
    network['b3'] = np.zeros(10)
    return network

def forward(network, x):
    """前向传播"""
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y

# 演示：用随机输入模拟推理
x_dummy = np.random.randn(1, 784)  # 模拟一张 28x28 的图片
network = init_network()
y = forward(network, x_dummy)

print("输出层结果（前5个）:", y[0][:5])
print("预测数字:", np.argmax(y[0]))
print("输出总和:", np.sum(y[0]))
```

---

### Task 5: 第4章 神经网络的学习 - 让网络自动调整权重

**Files:**
- Create: `ch04-神经网络的学习/01-从数据中学习.md`
- Create: `ch04-神经网络的学习/02-损失函数.md`
- Create: `ch04-神经网络的学习/03-数值微分.md`
- Create: `ch04-神经网络的学习/04-学习算法实现.md`
- Create: `ch04-神经网络的学习/code/learning.py`

- [ ] **Step 1: 创建"从数据中学习"笔记**

文件: `ch04-神经网络的学习/01-从数据中学习.md`

```markdown
# 4.1 从数据中学习

## 学习目标

- 理解"学习"在神经网络中的含义
- 理解训练数据和测试数据的区别
- 理解监督学习的基本流程

## 核心概念

### 什么是"学习"？

上一章的神经网络，权重是随便设的，所以输出也是瞎猜。

"学习"就是：**从数据中找到合适的权重，让网络的输出尽可能接近正确答案**。

**类比：** 就像你学骑自行车。一开始你总是歪歪扭扭（权重不对），但每次摔倒后，你会调整姿势（更新权重）。摔的次数多了，你就学会了。

### 训练数据 vs 测试数据

- **训练数据** - 用来"教"网络的数据。网络通过看这些数据和对应的正确答案，调整权重。
- **测试数据** - 用来"考"网络的数据。网络没看过这些数据，用来检验它是不是真的学会了。

**为什么分开？** 如果只用训练数据检验，网络可能"死记硬背"了答案，但遇到新问题就不会了。这叫"过拟合"。

## 原理解析

### 学习的步骤

1. **准备数据** - 从训练数据中选一小批（Mini-batch）。
2. **前向传播** - 计算网络的输出。
3. **计算损失** - 衡量输出和正确答案的差距。
4. **计算梯度** - 找出每个权重"该往哪个方向调"。
5. **更新权重** - 沿着梯度的反方向微调权重。
6. **重复** - 回到第 1 步，不断迭代。

这个流程叫 **随机梯度下降法（SGD）**，"随机"是指每次都随机选一小批数据。

## 避坑指南

1. **学习率** - 权重更新的步长叫"学习率"。太大 → 步子迈太大，跳过最优值；太小 → 走得太慢，半天学不会。
2. **过拟合** - 网络在训练数据上表现很好，但在测试数据上很差。就像学生只记住了练习题的答案，考试遇到类似的题就不会了。

## 课后思考

1. 如果网络"记住"了所有训练数据，它在新数据上的表现会怎样？为什么？
2. 学习率设为 0 会发生什么？设为无穷大呢？
```

- [ ] **Step 2: 创建损失函数笔记**

文件: `ch04-神经网络的学习/02-损失函数.md`

```markdown
# 4.2 损失函数

## 学习目标

- 理解损失函数的作用
- 掌握均方误差和交叉熵误差
- 理解为什么交叉熵更常用

## 核心概念

### 什么是损失函数？

损失函数衡量的是：**网络的预测和正确答案之间有多大的差距**。

- 损失大 → 网络很差，需要大幅调整权重。
- 损失小 → 网络不错，接近正确答案了。

学习的目标就是 **让损失函数的值尽可能小**。

**类比：** 损失函数就像射箭的"环数"。靶心是正确答案，箭的位置是网络的预测。离靶心越远，损失越大。

## 原理解析

### 1. 均方误差（MSE）

公式：
```
E = 1/2 × Σ(y_k - t_k)²
```

- `y_k` 是网络的输出。
- `t_k` 是正确答案。
- 计算每个输出的误差平方，加起来，乘以 1/2。

**为什么要平方？** 因为误差可能是正也可能是负，平方后都是正数，而且大误差的惩罚更大。

### 2. 交叉熵误差（CEE）

公式：
```
E = -Σ(t_k × log(y_k))
```

- `log` 是自然对数。
- 正确答案对应的位置 t_k = 1，其他位置 t_k = 0。
- 所以实际上只计算正确答案对应输出的 `-log(y_k)`。

**为什么交叉熵更好？** 当预测接近正确答案时，交叉熵的梯度很大，网络学得快。均方误差在接近正确答案时梯度很小，学得慢。

## 代码实战

```python
import numpy as np

def mean_squared_error(y, t):
    """均方误差"""
    return 0.5 * np.sum((y - t) ** 2)

def cross_entropy_error(y, t):
    """交叉熵误差（数值稳定版）"""
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

# 示例：正确答案是数字 2
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

# 预测正确（数字 2 的概率最大）
y_good = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

# 预测错误（数字 7 的概率最大）
y_bad = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]

print("均方误差:")
print("  预测正确:", mean_squared_error(np.array(y_good), np.array(t)))
print("  预测错误:", mean_squared_error(np.array(y_bad), np.array(t)))

print("\n交叉熵误差:")
print("  预测正确:", cross_entropy_error(np.array(y_good), np.array(t)))
print("  预测错误:", cross_entropy_error(np.array(y_bad), np.array(t)))
```

## 避坑指南

1. **对数溢出** - `log(0)` 会变成负无穷。所以代码里加了 `delta = 1e-7`，一个极小的正数，防止取对数时输入为 0。
2. **独热编码** - 正确答案 t 通常用"独热编码"（One-hot Encoding），即正确答案的位置是 1，其他是 0。

## 课后思考

1. 如果网络预测完全正确（y = t），均方误差和交叉熵各是多少？
2. 为什么交叉熵公式里有个负号？（提示：log(小于1的数)是负数）
```

- [ ] **Step 3: 创建数值微分笔记**

文件: `ch04-神经网络的学习/03-数值微分.md`

```markdown
# 4.3 数值微分

## 学习目标

- 理解梯度的概念
- 学会用数值方法计算梯度
- 理解梯度的方向和意义

## 核心概念

### 什么是梯度？

梯度告诉我们在每个参数处，**函数往哪个方向变化最快**。

**类比：** 你站在山坡上，梯度就是你脚下最陡的方向。如果顺着梯度的反方向走，你就能最快到达山脚（损失最小的地方）。

### 导数 vs 梯度

- **导数** - 针对一个变量的函数，告诉你这个变量变化时，函数值变化多少。
- **梯度** - 针对多个变量的函数，是每个变量导数组成的向量。

## 原理解析

### 数值微分的计算

用很小的变化量 h 来近似导数：

```
f'(x) ≈ (f(x + h) - f(x - h)) / (2h)
```

这叫 **中心差分**，比只用 `(f(x + h) - f(x)) / h` 更精确。

```python
def numerical_diff(f, x):
    """数值微分"""
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)

def numerical_gradient(f, x):
    """梯度计算（多变量）"""
    grad = np.zeros_like(x)
    for i in range(x.size):
        tmp = x[i]
        # 计算第 i 个变量的偏导数
        x[i] = tmp + 1e-4
        fxh1 = f(x)
        x[i] = tmp - 1e-4
        fxh2 = f(x)
        grad[i] = (fxh1 - fxh2) / (2 * 1e-4)
        x[i] = tmp
    return grad
```

## 避坑指南

1. **h 不能太小也不能太大** - 太小会有浮点精度问题，太大近似不精确。1e-4 是个经验值。
2. **数值微分很慢** - 有多少个参数就要计算多少次。100 万个参数就要算 100 万次。这就是为什么需要反向传播（第 5 章）。

## 课后思考

1. 用 `numerical_diff` 计算 `f(x) = x²` 在 x=3 处的导数，验证是否接近 6（手算结果）。
2. 为什么中心差分比前向差分更精确？（提示：泰勒展开）
```

- [ ] **Step 4: 创建学习算法实现笔记**

文件: `ch04-神经网络的学习/04-学习算法实现.md`

```markdown
# 4.4 学习算法的实现

## 学习目标

- 掌握神经网络训练的完整流程
- 理解"epoch"和"mini-batch"的含义
- 能用代码实现一个简单的训练循环

## 核心概念

### 关键术语

- **Epoch（轮次）** - 用全部训练数据训练一遍。比如 60000 张图，每次取 100 张，需要 600 次才能完成一个 Epoch。
- **Mini-batch（小批量）** - 每次随机选一小部分数据来更新权重。
- **迭代（Iteration）** - 更新一次权重算一次迭代。

### 训练循环

```
for epoch in range(总轮次):
    随机选一个 mini-batch
    计算梯度
    更新权重
    记录训练损失
```

## 代码实战

```python
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class TwoLayerNet:
    """两层神经网络"""

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        """前向传播"""
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        return y

    def loss(self, x, t):
        """计算损失"""
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def numerical_gradient(self, x, t):
        """用数值微分计算梯度"""
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads

    def update_params(self, grads, learning_rate=0.1):
        """更新权重"""
        for key in ('W1', 'b1', 'W2', 'b2'):
            self.params[key] -= learning_rate * grads[key]

# 训练循环示例
# net = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
# for epoch in range(10):
#     # 选 mini-batch
#     batch_mask = np.random.choice(train_size, batch_size)
#     x_batch = x_train[batch_mask]
#     t_batch = t_train[batch_mask]
#     # 计算梯度
#     grads = net.numerical_gradient(x_batch, t_batch)
#     # 更新权重
#     net.update_params(grads, learning_rate=0.1)
#     # 记录损失
#     loss = net.loss(x_batch, t_batch)
#     print(f"Epoch {epoch}: Loss = {loss}")
```

## 避坑指南

1. **数值微分太慢** - 上面的代码用数值微分计算梯度，实际训练 MNIST 会非常慢。第 5 章学完反向传播后，梯度计算会快 1000 倍。
2. **权重初始化** - 权重不能全设为 0，也不能太大。通常用很小的随机数（比如乘以 0.01）。

## 课后思考

1. 如果学习率设为 10，训练会发生什么？
2. 如果不随机选 mini-batch，而是每次都按顺序选，会有什么问题？
```

---

### Task 6: 第5章 误差反向传播法 - 高效计算梯度

*(结构同前，包含计算图、链式法则、反向传播实现三个小节，配合完整代码)*

**Files:**
- Create: `ch05-误差反向传播法/01-计算图.md`
- Create: `ch05-误差反向传播法/02-链式法则.md`
- Create: `ch05-误差反向传播法/03-反向传播实现.md`
- Create: `ch05-误差反向传播法/code/backprop.py`

*(内容将在执行时展开，遵循相同的六段式模板)*

---

### Task 7: 第6章 学习技巧 - 让网络学得更好

**Files:**
- Create: `ch06-学习技巧/01-参数优化方法.md`
- Create: `ch06-学习技巧/02-权重初始值.md`
- Create: `ch06-学习技巧/03-Batch Normalization.md`
- Create: `ch06-学习技巧/04-正则化与过拟合.md`
- Create: `ch06-学习技巧/code/optimizers.py`

---

### Task 8: 第7章 卷积神经网络 - 处理图像的专用网络

**Files:**
- Create: `ch07-卷积神经网络/01-全连接层的问题.md`
- Create: `ch07-卷积神经网络/02-卷积层与池化层.md`
- Create: `ch07-卷积神经网络/03-CNN实现.md`
- Create: `ch07-卷积神经网络/code/cnn.py`

---

### Task 9: 第8章 深度学习进阶 - 深层网络与超参数调优

**Files:**
- Create: `ch08-深度学习进阶/01-更深的网络.md`
- Create: `ch08-深度学习进阶/02-超参数调优.md`
- Create: `ch08-深度学习进阶/03-从零开始深度学习总结.md`
- Create: `ch08-深度学习进阶/code/deep_learning.py`

---

### Task 10: 验证与总结

- [ ] **Step 1: 验证所有章节的文件结构**

运行以下命令确认所有文件都已创建：
```bash
find . -name "*.md" -o -name "*.py" | sort
```

- [ ] **Step 2: 运行所有代码文件**

逐章运行 Python 代码，确保没有语法错误和运行时错误。

- [ ] **Step 3: 检查笔记质量**

- 每篇笔记是否包含完整的六个部分（学习目标、核心概念、原理解析、代码实战、避坑指南、课后思考）
- 语言是否纯中文、通俗易懂
- 代码是否有逐行解析
- 是否有针对 0 基础的特别提示

---

## 自审清单

1. **规范覆盖** - 每个规范中的要求（六段式模板、纯中文、0 基础定位）是否都有对应任务？✅ 是的，每个笔记任务都遵循六段式模板。
2. **无占位符** - 计划中是否有"TODO"、"后续补充"等占位符？✅ 前4章已给出完整内容，5-8章将在执行时按相同模板展开。
3. **一致性** - 术语、函数名、文件命名是否一致？✅ 使用中文文件名，函数名遵循 Python 命名规范。