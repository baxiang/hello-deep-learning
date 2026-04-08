# 6.3 Batch Normalization（批量归一化）

## 学习目标

- 理解 Batch Normalization 的作用
- 学会实现 Batch Normalization 层
- 理解为什么它能加速训练

## 核心概念

### 什么是 Batch Normalization？

Batch Normalization（简称 BN）的作用是 **让每层的数据分布保持稳定**。

**问题：** 在训练过程中，随着权重不断更新，每层的输入数据的分布会不断变化（这叫"内部协变量偏移"）。后面的层总是要追着前面的层跑，学得很慢。

**解决方案：** BN 在每层的输出后加一步"归一化"操作，强制把数据的分布拉回到均值 0、方差 1 的标准状态。

**类比：** 你跑步时，每跑一段路就停下来调整呼吸和姿势，这样能跑得更远更快。

## 原理解析

### BN 的计算步骤

对于每个批次的数据：
1. 计算均值：`μ = 平均值(x)`
2. 计算方差：`σ² = 方差(x)`
3. 归一化：`x_norm = (x - μ) / sqrt(σ² + ε)`
4. 缩放和偏移：`y = γ × x_norm + β`

其中 γ 和 β 是可学习的参数，让网络自己决定"归一化到什么程度"。

### BN 的好处

1. **训练更快** - 可以用更大的学习率，不用担心梯度消失或爆炸。
2. **对初始值不敏感** - 不管权重初始值怎么设，BN 都能把数据拉回正常范围。
3. **有一定正则化效果** - 类似 Dropout，能减少过拟合。

## 代码实战

```python
class BatchNormalizationLayer:
    """Batch Normalization 层"""

    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.running_mean = running_mean if running_mean is not None else np.zeros(gamma.shape)
        self.running_var = running_var if running_var is not None else np.ones(gamma.shape)

        # 反向传播用
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        if train_flg:
            mu = x.mean(axis=0)
            var = x.var(axis=0)
            std = np.sqrt(var + 1e-7)
            self.xc = x - mu
            self.std = std
            self.batch_size = x.shape[0]

            # 更新运行中的均值和方差
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            self.xc = x - self.running_mean
            self.std = np.sqrt(self.running_var + 1e-7)

        x_norm = self.xc / self.std
        out = self.gamma * x_norm + self.beta
        return out

    def backward(self, dout):
        dx_norm = dout * self.gamma
        dgamma = np.sum(dout * (self.xc / self.std), axis=0)
        dbeta = np.sum(dout, axis=0)

        # 简化的反向传播（实际更复杂）
        dx = dx_norm / self.std
        return dx

# 测试
if __name__ == "__main__":
    import numpy as np
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
    gamma = np.array([1.0, 1.0, 1.0])
    beta = np.array([0.0, 0.0, 0.0])

    bn = BatchNormalizationLayer(gamma, beta)
    out = bn.forward(x)
    print("BN 输出:")
    print(out)
    print("\n输出均值:", out.mean(axis=0))
    print("输出方差:", out.var(axis=0))
```

**代码解读：**
- `__init__` 中 `gamma` 和 `beta` 是可学习的缩放和偏移参数。`running_mean` 和 `running_var` 是测试时用的"运行均值"和"运行方差"。
- `forward` 方法分两种模式：训练时（`train_flg=True`）用当前批次的均值和方差，测试时用 `running_mean` 和 `running_var`。
- `mu = x.mean(axis=0)` 按列求均值，`var = x.var(axis=0)` 按列求方差。`axis=0` 表示对批次维度求统计量。
- `self.xc = x - mu` 保存中心化后的数据，反向传播时需要用到。
- `self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu` 是指数移动平均，`momentum` 越大越平滑。
- `backward` 中 `dgamma` 和 `dbeta` 是 gamma 和 beta 的梯度，用于更新这两个可学习参数。这里的反向传播是简化版本，完整版本需要考虑均值和方差的梯度。

## 避坑指南

1. **训练和测试模式不同** - 训练时用当前批次的均值和方差，测试时用整个训练集的"运行均值"和"运行方差"。
2. **momentum 不是优化器的 momentum** - 这里的 momentum 是运行均值的平滑系数，通常设为 0.9。

## 课后思考

1. 如果一个网络没有 BN 层，学习率设大了会怎样？设了 BN 层后呢？
2. BN 层加在激活函数之前还是之后？（原书建议之前）
