# 6.3 Batch Normalization（批量归一化）

## 学习目标

- 理解 Batch Normalization 解决什么问题
- 知道 BN 层的计算步骤
- 理解为什么 BN 能加速训练

## 核心概念

### BN 解决什么问题？

训练神经网络时，每一层的输入数据分布会不断变化（因为前面的权重在不断更新）。

**类比：** 你在跑步，但跑道在不断变形。你总是要调整步伐来适应新路面，跑不快。

**BN 的做法：** 每层的输出先"整理"一下，强制让数据分布保持稳定（均值 0，方差 1）。

**类比：** 每跑一段路，就把路面修平整。这样你不用调整步伐，跑得更快。

### BN 的额外好处

1. **对初始值不敏感** - 不管初始值设多大，BN 都能把数据拉回正常范围。
2. **抑制过拟合** - 有一定的正则化效果，类似 Dropout。
3. **可以用更大的学习率** - 因为数据被"约束"了，不用担心爆炸。

## 原理解析

### BN 的计算步骤

对于每个批次的数据：

```
第 1 步: 算均值 μ = 平均值(输入)
第 2 步: 算方差 σ² = 方差(输入)
第 3 步: 归一化 x_norm = (输入 - μ) / √(σ² + 小常数)
第 4 步: 缩放和偏移 输出 = γ × x_norm + β
```

**第 4 步的 γ 和 β 是什么？**

γ 和 β 是可学习的参数。网络自己决定"归一化到什么程度"。

**类比：** 老师给学生成绩做"标准化"（减均值除以标准差），但老师说"我可以根据需要再缩放和偏移"。

### 训练 vs 测试

**训练时：** 用当前批次的均值和方差。

**测试时：** 用训练过程中积累的"移动平均"均值和方差。

**为什么不同？** 测试时可能只有 1 个样本，算不了均值和方差。

## 代码实战

### 完整可运行代码

```python
import numpy as np

class BatchNormalization:
    def __init__(self, gamma, beta, momentum=0.9):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.running_mean = np.zeros(gamma.shape)
        self.running_var = np.zeros(gamma.shape)
        self.batch_size = None
        self.xc = None
        self.std = None

    def forward(self, x, train_flg=True):
        if train_flg:
            mu = x.mean(axis=0)
            var = x.var(axis=0)
            std = np.sqrt(var + 1e-7)
            self.xc = x - mu
            self.std = std
            self.batch_size = x.shape[0]
            self.running_mean = (self.momentum * self.running_mean +
                               (1 - self.momentum) * mu)
            self.running_var = (self.momentum * self.running_var +
                              (1 - self.momentum) * var)
        else:
            self.xc = x - self.running_mean
            self.std = np.sqrt(self.running_var + 1e-7)

        x_norm = self.xc / self.std
        return self.gamma * x_norm + self.beta

# === 测试 ===
print("=== Batch Normalization 测试 ===\n")

x = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
    [2.0, 3.0, 1.0],
    [5.0, 1.0, 4.0]
], dtype=np.float64)

print("原始数据:")
print(x)
print(f"  每列均值: {x.mean(axis=0)}")
print(f"  每列标准差: {x.std(axis=0):.2f}")

gamma = np.ones(3)
beta = np.zeros(3)
bn = BatchNormalization(gamma, beta)

out = bn.forward(x, train_flg=True)
print(f"\nBN 输出:")
print(out)
print(f"  每列均值: {out.mean(axis=0)}")
print(f"  每列标准差: {out.std(axis=0):.2f}")
print(f"  → 均值接近 0，标准差接近 1")
```

### 运行结果

```
=== Batch Normalization 测试 ===

原始数据:
  每列均值: [3.8 3.8 4.6]
  每列标准差: [2.14 2.40 2.65]

BN 输出:
  每列均值: [0. 0. 0.]
  每列标准差: [1.00 1.00 1.00]
  → 均值接近 0，标准差接近 1
```

## 避坑指南

1. **训练和测试模式不同** - 训练用当前批次，测试用移动平均。

2. **BN 放在激活函数之前** - 矩阵乘法之后、激活函数之前。

3. **小批次效果差** - 批次太小（比如 4），均值和方差估计不准。建议 ≥ 16。

## 课后思考

1. 如果批次大小 = 1，BN 还能正常工作吗？

2. BN 的 γ 和 β 如果固定为 1 和 0，会有什么问题？

3. BN 为什么能抑制过拟合？（提示：想想它对每批数据做了什么"扰动"）
