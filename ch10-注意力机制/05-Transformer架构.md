# 10.5 Transformer 架构

## 学习目标

- 理解 Transformer 的整体结构
- 掌握 Encoder-Decoder 架构
- 了解残差连接和层归一化的作用

## 核心概念

### Transformer：注意力革命

2017 年，Google 发表论文《Attention Is All You Need》，提出了 Transformer。

**核心思想：完全用注意力机制，不用RNN 和CNN。**

之前：RNN 是序列建模的主流，但有速度慢、长距离依赖问题
之后：Transformer 用注意力解决这些问题，成为主流

Transformer 影响：
- BERT、GPT 等预训练模型的基础
- ChatGPT 背后的技术
- 几乎所有现代 NLP 模型

### Encoder-Decoder 结构

Transformer 用于翻译任务，采用 Encoder-Decoder 结构：

```
原文 → Encoder → 编码表示 → Decoder →译文
```

**Encoder（编码器）：**
- 理解原文，提取信息
- 多层堆叠，每层有自注意力
- 输出是原文的"深层表示"

**Decoder（解码器）：**
- 根据编码表示，生成译文
- 多层堆叠，每层有自注意力 + 交叉注意力
- 自注意力：关注已生成的译文
- 交叉注意力：关注 Encoder 的输出

### Transformer 的关键组件

| 组件 | 作用 |
|------|------|
| 多头注意力 | 多角度关注信息 |
| 位置编码 | 补充位置信息 |
| 残差连接 | 让梯度更容易传播 |
| 层归一化 | 稳定训练 |
| 前馈网络 | 增加非线性变换 |

## 原理解析

### Encoder 层结构

每个 Encoder 层包含：

```
输入 → 多头注意力 → 残差连接 + LayerNorm → 前馈网络 → 殀差连接 + LayerNorm → 输出
```

**残差连接（Residual Connection）：**

```
输出 = 输入 + 子层输出
```

作用：
- 让梯度能"跳过"某些层直接传播
- 防止梯度消失
- 类似 ResNet 的设计

**层归一化（Layer Normalization）：**

```
归一化后的值 = (值 - 均值) / 标准差
```

作用：
- 稳定每一层的输入分布
- 加速训练收敛
- 与 BatchNorm 不同，LN 是对每个样本归一化

### Decoder 层结构

Decoder 层比 Encoder 多一个注意力：

```
输入 → 自注意力（掩码）→ Add & Norm → 交叉注意力 → Add & Norm → 前馈网络 → Add & Norm → 输出
```

**掩码自注意力：**

生成译文时，只能看到已经生成的词，不能看后面的词。

```
掩码矩阵：
    词1  词2  词3  词4
词1  0   -∞   -∞   -∞    ← 词1 只能看词1
词2  0    0   -∞   -∞    ← 词2 能看词1、词2
词3  0    0    0   -∞    ← 词3 能看词1-词3
词4  0    0    0    0    ← 词4 能看所有词
```

-∞ 经过 softmax 变成 0，实现"看不到"的效果。

**交叉注意力：**

Q 来自 Decoder（当前生成的词），K、V 来自 Encoder（原文信息）。

这让 Decoder 能"参考"原文。

### 前馈网络（FFN）

```
FFN(x) = max(0, x·W_1 + b_1)·W_2 + b_2
```

两层线性变换，中间有 ReLU。

作用：
- 增加非线性变换能力
- 让模型能学习更复杂的特征

## 代码实战

```python
import numpy as np


def softmax(x, axis=-1):
    """Softmax 函数"""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def layer_norm(x, eps=1e-6):
    """层归一化"""
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + eps)


class TransformerBlock:
    """简化版 Transformer 块"""

    def __init__(self, d_model, num_heads, d_ff):
        """初始化

        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 注意力权重
        np.random.seed(42)
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        self.W_o = np.random.randn(d_model, d_model) * 0.1

        # 前馈网络权重
        self.W_1 = np.random.randn(d_model, d_ff) * 0.1
        self.b_1 = np.zeros(d_ff)
        self.W_2 = np.random.randn(d_ff, d_model) * 0.1
        self.b_2 = np.zeros(d_model)

    def multi_head_attention(self, X, mask=None):
        """多头注意力"""
        seq_len = X.shape[0]

        Q = np.dot(X, self.W_q)
        K = np.dot(X, self.W_k)
        V = np.dot(X, self.W_v)

        # 分头
        Q_heads = Q.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)
        K_heads = K.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)
        V_heads = V.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)

        # 各头计算
        outputs = []
        for h in range(self.num_heads):
            scores = np.dot(Q_heads[h], K_heads[h].T) / np.sqrt(self.d_k)

            if mask is not None:
                scores = scores + mask

            weights = softmax(scores)
            head_out = np.dot(weights, V_heads[h])
            outputs.append(head_out)

        # 拼接
        concat = np.concatenate(outputs, axis=-1)
        return np.dot(concat, self.W_o)

    def feed_forward(self, X):
        """前馈网络"""
        hidden = np.maximum(0, np.dot(X, self.W_1) + self.b_1)  # ReLU
        return np.dot(hidden, self.W_2) + self.b_2

    def forward(self, X, mask=None):
        """Transformer 块前向传播"""
        # 多头注意力 + 残差 + LayerNorm
        attn_out = self.multi_head_attention(X, mask)
        X = layer_norm(X + attn_out)

        # 前馈网络 + 残差 + LayerNorm
        ff_out = self.feed_forward(X)
        X = layer_norm(X + ff_out)

        return X


class SimpleTransformer:
    """简化版 Transformer"""

    def __init__(self, d_model=64, num_heads=4, d_ff=256, num_layers=2):
        """初始化

        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络维度
            num_layers: 层数
        """
        self.d_model = d_model
        self.num_layers = num_layers

        # Encoder 层
        self.encoder_layers = [
            TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)
        ]

        # Decoder 层（简化，省略交叉注意力）
        self.decoder_layers = [
            TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)
        ]

    def create_causal_mask(self, seq_len):
        """创建因果掩码（Decoder 用）"""
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        mask = mask * -1e9  # 上三角变成 -∞
        return mask

    def encode(self, X):
        """编码"""
        for layer in self.encoder_layers:
            X = layer.forward(X)
        return X

    def decode(self, X, encoder_output):
        """解码（简化版，省略交叉注意力）"""
        seq_len = X.shape[0]
        mask = self.create_causal_mask(seq_len)

        for layer in self.decoder_layers:
            X = layer.forward(X, mask)

        return X

    def forward(self, src, tgt):
        """完整前向传播"""
        encoded = self.encode(src)
        decoded = self.decode(tgt, encoded)
        return decoded


if __name__ == "__main__":
    print("=" * 50)
    print("Transformer 架构演示")
    print("=" * 50)

    np.random.seed(42)

    # 创建 Transformer
    d_model = 16
    num_heads = 4
    d_ff = 32
    num_layers = 2

    transformer = SimpleTransformer(d_model, num_heads, d_ff, num_layers)

    print(f"\nTransformer 配置:")
    print(f"  模型维度: {d_model}")
    print(f"  注意力头数: {num_heads}")
    print(f"  前馈网络维度: {d_ff}")
    print(f"  层数: {num_layers}")

    # 模拟输入
    src_len = 4  # 原文长度
    tgt_len = 3  #译文长度

    src = np.random.randn(src_len, d_model)
    tgt = np.random.randn(tgt_len, d_model)

    print(f"\n原文向量形状: {src.shape}")
    print(f"译文向量形状: {tgt.shape}")

    # 编码
    encoded = transformer.encode(src)
    print(f"\nEncoder 输出形状: {encoded.shape}")

    # 解码
    decoded = transformer.decode(tgt, encoded)
    print(f"Decoder 输出形状: {decoded.shape}")

    # 展示因果掩码
    print("\n=== 因果掩码（Decoder 自注意力用）===")
    mask = transformer.create_causal_mask(4)
    print("掩码矩阵（-1e9 表示看不见）:")
    print(mask)

    print("\n经过 softmax 后的效果:")
    masked_scores = np.ones((4, 4)) + mask
    print(softmax(masked_scores))

    print("\n" + "=" * 50)
    print("Transformer 关键点总结:")
    print("1. Encoder: 多层自注意力 + 前馈网络")
    print("2. Decoder: 自注意力（掩码）+ 交叉注意力 + 前馈网络")
    print("3. 残差连接: 输入 + 子层输出，梯度更容易传播")
    print("4. 层归一化: 稳定训练，加速收敛")
    print("5. 完全不用 RNN，并行计算效率高")
    print("=" * 50)