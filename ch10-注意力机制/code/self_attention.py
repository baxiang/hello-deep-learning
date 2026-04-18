"""
第10章 注意力机制 - 自注意力实现
包含：自注意力模块、位置编码、掩码
"""

import numpy as np


def softmax(x, axis=-1):
    """Softmax 函数"""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def positional_encoding(seq_len, d_model):
    """位置编码（Transformer 原版）

    公式:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        seq_len: 序列长度
        d_model: 模型维度

    Returns:
        PE: 位置编码矩阵，形状 (seq_len, d_model)
    """
    PE = np.zeros((seq_len, d_model))

    for pos in range(seq_len):
        for i in range(d_model):
            if i % 2 == 0:
                PE[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            else:
                PE[pos, i] = np.cos(pos / (10000 ** ((i - 1) / d_model)))

    return PE


def create_attention_mask(seq_len, mask_type="causal"):
    """创建注意力掩码

    Args:
        seq_len: 序列长度
        mask_type: 掩码类型
            - "causal": 因果掩码（只能看自己和前面的）
            - "padding": 填充掩码（忽略填充位置）

    Returns:
        mask: 掩码矩阵
    """
    if mask_type == "causal":
        # 上三角为 -inf，下三角为 0
        mask = np.triu(np.ones((seq_len, seq_len)) * -1e9, k=1)
    elif mask_type == "padding":
        # 假设最后几个位置是填充（示例）
        mask = np.zeros((seq_len, seq_len))
        # 假设后 2 个位置是填充
        mask[:, -2:] = -1e9
    else:
        mask = np.zeros((seq_len, seq_len))

    return mask


class SelfAttention:
    """自注意力模块

    自注意力：Q、K、V 都来自同一个序列
    让序列中的每个元素都能关注其他所有元素
    """

    def __init__(self, d_model):
        """初始化

        Args:
            d_model: 模型维度
        """
        self.d_model = d_model

        # 初始化 QKV 投影权重
        np.random.seed(42)
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1

    def forward(self, X, mask=None):
        """前向传播

        Args:
            X: 输入序列，形状 (seq_len, d_model)
            mask: 可选掩码矩阵

        Returns:
            output: 自注意力输出
            weights: 注意力权重矩阵
        """
        # 计算 Q、K、V
        Q = np.dot(X, self.W_q)  # (seq_len, d_model)
        K = np.dot(X, self.W_k)
        V = np.dot(X, self.W_v)

        # 计算注意力分数
        scores = np.dot(Q, K.T) / np.sqrt(self.d_model)

        # 应用掩码
        if mask is not None:
            scores = scores + mask

        # softmax
        weights = softmax(scores, axis=-1)

        # 加权求和
        output = np.dot(weights, V)

        return output, weights

    def __call__(self, X, mask=None):
        return self.forward(X, mask)


class SelfAttentionWithProjection:
    """带输出投影的自注意力"""

    def __init__(self, d_model):
        self.d_model = d_model
        self.self_attn = SelfAttention(d_model)

        # 输出投影
        np.random.seed(42)
        self.W_o = np.random.randn(d_model, d_model) * 0.1

    def forward(self, X, mask=None):
        """前向传播"""
        attn_output, weights = self.self_attn(X, mask)
        output = np.dot(attn_output, self.W_o)
        return output, weights


if __name__ == "__main__":
    print("=" * 50)
    print("自注意力演示")
    print("=" * 50)

    np.random.seed(42)

    # === 1. 位置编码 ===
    print("\n=== 1. 位置编码 ===")

    seq_len = 6
    d_model = 8

    PE = positional_encoding(seq_len, d_model)

    print(f"位置编码形状: {PE.shape}")
    print("\n不同位置的编码:")
    for pos in range(seq_len):
        print(f" 位置{pos}: 前4维 {PE[pos, :4]}")

    print("\n位置编码特点:")
    print("  - 每个位置有唯一的编码")
    print("  - 相邻位置编码相似（sin/cos 连续变化）")
    print("  - 可以泛化到任意长度")

    # === 2. 自注意力基本示例 ===
    print("\n=== 2. 自注意力计算 ===")

    # 模拟句子："我 喜欢 深度 学习"
    words = ["我", "喜欢", "深度", "学习"]
    seq_len = len(words)
    d_model = 8

    X = np.random.randn(seq_len, d_model)

    self_attn = SelfAttention(d_model)
    output, weights = self_attn.forward(X)

    print(f"输入形状: {X.shape}")
    print(f"输出形状: {output.shape}")

    print("\n注意力权重矩阵:")
    print("      ", end="")
    for w in words:
        print(f"{w:6s}", end=" ")
    print()
    for i, w in enumerate(words):
        print(f"{w:4s} ", end="")
        for j in range(seq_len):
            print(f"{weights[i, j]:.3f} ", end="")
        print()

    # === 3. 加入位置编码 ===
    print("\n=== 3. 带位置编码的自注意力 ===")

    PE = positional_encoding(seq_len, d_model)
    X_with_pos = X + PE

    output_pos, weights_pos = self_attn.forward(X_with_pos)

    print("加入位置编码后的注意力权重:")
    print("      ", end="")
    for w in words:
        print(f"{w:6s}", end=" ")
    print()
    for i, w in enumerate(words):
        print(f"{w:4s} ", end="")
        for j in range(seq_len):
            print(f"{weights_pos[i, j]:.3f} ", end="")
        print()

    print("\n位置编码的影响:")
    print("  - 让模型能区分词的顺序")
    print("  - '我喜欢'和'喜欢我'会产生不同的注意力")

    # === 4. 因果掩码 ===
    print("\n=== 4. 因果掩码（用于生成任务）===")

    mask = create_attention_mask(seq_len, "causal")

    print("因果掩码矩阵:")
    print(mask)

    output_masked, weights_masked = self_attn.forward(X, mask)

    print("\n应用因果掩码后的注意力:")
    print("      ", end="")
    for w in words:
        print(f"{w:6s}", end=" ")
    print()
    for i, w in enumerate(words):
        print(f"{w:4s} ", end="")
        for j in range(seq_len):
            print(f"{weights_masked[i, j]:.3f} ", end="")
        print()

    print("\n因果掩码的作用:")
    print("  - 每个词只能看自己和前面的词")
    print("  - 用于生成任务（翻译、文本生成）")
    print("  - 保证生成顺序的正确性")

    print("\n" + "=" * 50)
    print("自注意力总结:")
    print("1. Q、K、V 来自同一序列")
    print("2. 每个词能直接关注任意其他词")
    print("3. 位置编码补充顺序信息")
    print("4. 因果掩码用于生成任务")
    print("=" * 50)
