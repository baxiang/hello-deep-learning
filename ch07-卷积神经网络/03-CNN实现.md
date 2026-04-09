# 7.3 完整的 CNN 实现

## 学习目标

- 把卷积层、池化层、全连接层组合起来
- 理解 CNN 的数据流
- 实现一个处理 MNIST 的简单 CNN

## 核心概念

### CNN 的典型结构

```
输入 → 卷积 → ReLU → 池化 → 全连接 → 输出
```

- **前半部分**（卷积 + 池化）：提取特征
- **后半部分**（全连接）：根据特征做分类

### 数据流

以 MNIST（28×28 单通道）为例：

```
输入: (10, 1, 28, 28)  ← 10 张图片, 1 通道, 28×28
↓ 卷积 (10 个 5×5 滤波器)
(10, 10, 24, 24)       ← 10 个通道（10 个滤波器）
↓ 池化 (2×2, 步幅 2)
(10, 10, 12, 12)       ← 尺寸减半
↓ 展平
(10, 1440)             ← 10 × 12 × 12 = 1440
↓ 全连接
(10, 10)               ← 10 个类别的概率
```

## 代码实战

### 完整可运行代码

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

def softmax(a):
    c = np.max(a, axis=1, keepdims=True)
    return np.exp(a - c) / np.sum(np.exp(a - c), axis=1, keepdims=True)

def conv_simple(x, w, b, stride=1, pad=0):
    """简化卷积: x (N,C,H,W), w (FN,C,FH,FW)"""
    N, C, H, W = x.shape
    FN, _, FH, FW = w.shape
    out_h = (H + 2*pad - FH) // stride + 1
    out_w = (W + 2*pad - FW) // stride + 1

    out = np.zeros((N, FN, out_h, out_w))
    if pad > 0:
        x = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)))

    for i in range(out_h):
        for j in range(out_w):
            xs = x[:, :, i:i+FH, j:j+FW]
            for f in range(FN):
                out[:, f, i, j] = np.sum(xs * w[f], axis=(1,2,3)) + b[f]
    return out

def max_pooling_simple(x, pool=2, stride=2):
    """简化最大池化"""
    N, C, H, W = x.shape
    out_h = (H - pool) // stride + 1
    out_w = (W - pool) // stride + 1
    out = np.zeros((N, C, out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            xs = x[:, :, i:i+pool, j:j+pool]
            out[:, :, i, j] = np.max(xs, axis=(2,3))
    return out

# === 创建简单 CNN ===
np.random.seed(42)

# 参数
filter_num = 10
filter_size = 5
input_size = 784  # 28×28
hidden_size = 50
output_size = 10

# 初始化权重
W1 = 0.01 * np.random.randn(filter_num, 1, filter_size, filter_size)
b1 = np.zeros(filter_num)
W2 = 0.01 * np.random.randn(filter_num * 12 * 12, hidden_size)
b2 = np.zeros(hidden_size)
W3 = 0.01 * np.random.randn(hidden_size, output_size)
b3 = np.zeros(output_size)

print("=== 简单 CNN 前向传播 ===\n")

# 模拟 5 张图片
x = np.random.randn(5, 1, 28, 28)
print(f"输入形状: {x.shape}")

# 卷积 → ReLU → 池化
conv_out = conv_simple(x, W1, b1)
print(f"卷积后: {conv_out.shape}")

relu_out = relu(conv_out)
print(f"ReLU 后: {relu_out.shape}")

pool_out = max_pooling_simple(relu_out)
print(f"池化后: {pool_out.shape}")

# 展平
N = pool_out.shape[0]
flat = pool_out.reshape(N, -1)
print(f"展平后: {flat.shape}")

# 全连接 → Softmax
a2 = np.dot(flat, W2) + b2
z2 = relu(a2)
a3 = np.dot(z2, W3) + b3
y = softmax(a3)

print(f"\n输出形状: {y.shape}")
print("预测结果:")
for i in range(5):
    pred = np.argmax(y[i])
    print(f"  图片 {i}: 预测数字 = {pred}, 概率 = {y[i][pred]:.2%}")
```

### 运行结果

```
=== 简单 CNN 前向传播 ===

输入形状: (5, 1, 28, 28)
卷积后: (5, 10, 24, 24)
ReLU 后: (5, 10, 24, 24)
池化后: (5, 10, 12, 12)
展平后: (5, 1440)

输出形状: (5, 10)
预测结果:
  图片 0: 预测数字 = 5, 概率 = 10.05%
  图片 1: 预测数字 = 8, 概率 = 10.03%
  ...
```

**解读：**
- 因为权重是随机的，每个数字的概率差不多（约 10%）。
- 但流程是对的：卷积提取特征 → 池化缩小 → 全连接分类。
- 训练之后，正确答案的概率会接近 100%。

## 避坑指南

1. **四维数组** - CNN 数据是 `(批次, 通道, 高, 宽)`，不是二维的。

2. **展平时机** - 只能在池化后展平。展平后才能接全连接层。

3. **卷积很慢** - 这个代码用 for 循环，实际中用 im2col 等技巧加速。

## 课后思考

1. 如果滤波器数从 10 改成 20，展平后的向量有多大？

2. 为什么卷积层的输出通道数 = 滤波器个数？

3. 如果不做池化，直接接全连接层，参数会增加多少？
