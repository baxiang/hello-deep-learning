# 7.3 完整的 CNN 实现

## 学习目标

- 把卷积层、池化层、全连接层组合起来
- 理解 CNN 的数据流
- 实现一个可以处理 MNIST 的简单 CNN

## 核心概念

### CNN 的典型结构

```
输入 → 卷积 → ReLU → 池化 → 卷积 → ReLU → 池化 → 全连接 → ReLU → 全连接 → Softmax → 输出
```

- **前半部分**（卷积 + 池化）：提取特征。从低级特征（边缘、角点）到高级特征（眼睛、轮子）。
- **后半部分**（全连接）：根据提取的特征做分类。

## 原理解析

### 数据流

以 MNIST（28×28 单通道）为例：

```
输入: (1, 28, 28, 1)  ← (批次, 高, 宽, 通道)
↓ 卷积 (32 个 5×5 滤波器, 步幅 1, 填充 0)
(1, 24, 24, 32)
↓ ReLU
(1, 24, 24, 32)
↓ 最大池化 (2×2, 步幅 2)
(1, 12, 12, 32)
↓ 展平
(1, 4608)  ← 12 × 12 × 32
↓ 全连接 (100 个节点)
(1, 100)
↓ ReLU
(1, 100)
↓ 全连接 (10 个节点)
(1, 10)
↓ Softmax
(1, 10)  ← 每个数字的概率
```

### 四维数组

CNN 中的数据是四维的：`(批次大小, 通道, 高, 宽)` 或 `(批次大小, 高, 宽, 通道)`。

NumPy 中通常用第二种（通道在最后）。

## 代码实战

```python
import numpy as np
import sys
sys.path.append('..')

def relu(x):
    return np.maximum(0, x)

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    return exp_a / np.sum(exp_a)

class SimpleConvNet:
    """简单的卷积神经网络"""

    def __init__(self, input_dim=(1, 28, 28), 
                 conv_param={'filter_num': 30, 'filter_size': 5, 'stride': 1, 'pad': 0},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_stride = conv_param['stride']
        filter_pad = conv_param['pad']
        
        input_c = input_dim[0]
        input_h = input_dim[1]
        input_w = input_dim[2]
        
        # 卷积后的大小
        conv_out_h = (input_h + 2 * filter_pad - filter_size) // filter_stride + 1
        conv_out_w = (input_w + 2 * filter_pad - filter_size) // filter_stride + 1
        
        # 池化后的大小
        pool_out_h = conv_out_h // 2
        pool_out_w = conv_out_w // 2
        
        # 全连接层的输入大小
        fc_input_size = pool_out_h * pool_out_w * filter_num
        
        # 初始化权重
        self.params = {}
        # 卷积层权重 (filter_num, input_c, filter_size, filter_size)
        self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_c, filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        # 全连接层权重
        self.params['W2'] = weight_init_std * np.random.randn(fc_input_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)
        
        # 保存参数
        self.filter_num = filter_num
        self.filter_size = filter_size
        self.filter_stride = filter_stride
        self.filter_pad = filter_pad
        self.pool_size = 2
        self.pool_stride = 2
        self.fc_input_size = fc_input_size

    def conv_simple(self, x, w, b, stride, pad):
        """简化的卷积操作"""
        n, c, h, w_x = x.shape
        f_n, f_c, f_h, f_w = w.shape
        out_h = (h + 2 * pad - f_h) // stride + 1
        out_w = (w_x + 2 * pad - f_w) // stride + 1
        
        out = np.zeros((n, f_n, out_h, out_w))
        x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
        
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride
                w_start = j * stride
                x_slice = x_padded[:, :, h_start:h_start+f_h, w_start:w_start+f_w]
                for f in range(f_n):
                    out[:, f, i, j] = np.sum(x_slice * w[f], axis=(1, 2, 3)) + b[f]
        return out

    def pool_simple(self, x, pool_size, pool_stride):
        """简化的最大池化"""
        n, c, h, w = x.shape
        out_h = (h - pool_size) // pool_stride + 1
        out_w = (w - pool_size) // pool_stride + 1
        
        out = np.zeros((n, c, out_h, out_w))
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * pool_stride
                w_start = j * pool_stride
                x_slice = x[:, :, h_start:h_start+pool_size, w_start:w_start+pool_size]
                out[:, :, i, j] = np.max(x_slice, axis=(2, 3))
        return out

    def predict(self, x):
        """前向传播"""
        W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']
        b1, b2, b3 = self.params['b1'], self.params['b2'], self.params['b3']
        
        # 卷积 → ReLU → 池化
        conv_out = self.conv_simple(x, W1, b1, self.filter_stride, self.filter_pad)
        relu_out = relu(conv_out)
        pool_out = self.pool_simple(relu_out, self.pool_size, self.pool_stride)
        
        # 展平
        n = pool_out.shape[0]
        flat = pool_out.reshape(n, -1)
        
        # 全连接 → ReLU → 全连接
        affine1 = np.dot(flat, W2) + b2
        relu2 = relu(affine1)
        affine2 = np.dot(relu2, W3) + b3
        
        # Softmax
        y = softmax(affine2)
        return y

if __name__ == "__main__":
    # 创建网络
    net = SimpleConvNet(input_dim=(1, 28, 28))
    
    # 随机输入模拟 10 张图片
    x_dummy = np.random.randn(10, 1, 28, 28)
    
    # 推理
    y = net.predict(x_dummy)
    print("输出形状:", y.shape)
    print("每张图片的预测:")
    for i in range(10):
        print(f"  图片 {i}: 预测数字 = {np.argmax(y[i])}, 概率 = {np.max(y[i]):.4f}")
    print("所有输出总和 (应≈1):", np.sum(y, axis=1))
```

## 避坑指南

1. **权重维度** - 卷积层权重是四维的 `(滤波器数, 输入通道, 滤波器高, 滤波器宽)`，不是二维的。
2. **展平的时机** - 展平只能在池化层之后做。展平后数据变成二维 `(批次, 特征数)`，才能接全连接层。
3. **卷积很慢** - 上面的代码用 for 循环实现卷积，非常慢。实际中会用 im2col 等技巧加速，或者直接用深度学习框架。

## 课后思考

1. 如果把卷积层的滤波器数从 30 改成 60，全连接层的输入大小会变成多少？
2. 为什么卷积层的输出通道数等于滤波器的个数？
3. 如果不做池化，直接接全连接层，会有什么问题？
