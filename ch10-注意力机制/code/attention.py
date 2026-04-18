"""
第10章 注意力机制 - 基础注意力计算
包含：点积注意力、加性注意力、QKV 框架
"""

import numpy as np


def softmax(x, axis=-1):
    """Softmax 函数

    Args:
        x: 输入数组
        axis: 计算softmax 的维度

    Returns:
        softmax 结果
    """
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """缩放点积注意力

    公式: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Args:
        Q: 查询矩阵，形状 (seq_len_q, d_k) 或(batch_size, seq_len_q, d_k)
        K: 键矩阵，形状 (seq_len_k, d_k) 或 (batch_size, seq_len_k, d_k)
        V: 值矩阵，形状 (seq_len_k, d_v) 或 (batch_size, seq_len_k, d_v)
        mask: 可选掩码矩阵

    Returns:
        output: 注意力输出
        weights: 注意力权重矩阵
    """
    d_k = Q.shape[-1]

    # 计算注意力分数: QK^T
    scores = np.dot(Q, K.swapaxes(-1, -2)) / np.sqrt(d_k)

    # 应用掩码（如果有）
    if mask is not None:
        scores = scores + mask  # mask 中 -inf的位置会被 softmax 变成 0

    # softmax 归一化
    weights = softmax(scores, axis=-1)

    # 加权求和: weights * V
    output = np.dot(weights, V)

    return output, weights


def additive_attention(Q, K, V, d_hidden=8):
    """加性注意力（Bahdanau 注意力）

    公式: scores = tanh(W_q Q + W_k K)
         weights = softmax(W_s scores)
         output = weights * V

    Args:
        Q: 查询矩阵，形状 (seq_len_q, d_k)
        K: 键矩阵，形状 (seq_len_k, d_k)
        V: 值矩阵，形状 (seq_len_k, d_v)
        d_hidden: 隐藏层维度

    Returns:
        output: 注意力输出
        weights: 注意力权重
    """
    seq_len_q = Q.shape[0]
    seq_len_k = K.shape[0]
    d_k = Q.shape[-1]

    # 初始化权重（模拟）
    np.random.seed(42)
    W_q = np.random.randn(d_k, d_hidden) * 0.1
    W_k = np.random.randn(d_k, d_hidden) * 0.1
    v = np.random.randn(d_hidden) * 0.1

    # 扩展 Q 和 K 以便计算
    # Q_expanded: (seq_len_q, seq_len_k, d_k)
    # K_expanded: (seq_len_q, seq_len_k, d_k)
    Q_expanded = np.tile(Q[:, np.newaxis, :], (1, seq_len_k, 1))
    K_expanded = np.tile(K[np.newaxis, :, :], (seq_len_q, 1, 1))

    # 计算 tanh(W_q * Q + W_k * K)
    hidden = np.tanh(
        np.dot(Q_expanded, W_q) + np.dot(K_expanded, W_k)
    )  # (seq_len_q, seq_len_k, d_hidden)

    # 计算分数: v^T * hidden
    scores = np.dot(hidden, v)  # (seq_len_q, seq_len_k)

    # softmax
    weights = softmax(scores, axis=-1)

    # 加权求和
    output = np.dot(weights, V)

    return output, weights


class AttentionLayer:
    """注意力层（封装点积注意力）"""

    def __init__(self, d_k, d_v):
        """初始化

        Args:
            d_k: 键维度
            d_v: 值维度
        """
        self.d_k = d_k
        self.d_v = d_v

    def forward(self, Q, K, V, mask=None):
        """前向传播"""
        return scaled_dot_product_attention(Q, K, V, mask)

    def __call__(self, Q, K, V, mask=None):
        return self.forward(Q, K, V, mask)


if __name__ == "__main__":
    print("=" * 50)
    print("注意力计算演示")
    print("=" * 50)

    np.random.seed(42)

    # === 1. 点积注意力基础示例 ===
    print("\n=== 1. 缩放点积注意力 ===")

    # 模拟数据
    seq_len_q = 2  # 查询数量
    seq_len_k = 4  # 输入数量
    d_k = 8  # 键维度
    d_v = 6  # 值维度

    Q = np.random.randn(seq_len_q, d_k)
    K = np.random.randn(seq_len_k, d_k)
    V = np.random.randn(seq_len_k, d_v)

    print(f"Q 形状: {Q.shape} (查询数={seq_len_q})")
    print(f"K 形状: {K.shape} (输入数={seq_len_k})")
    print(f"V 形状: {V.shape}")

    output, weights = scaled_dot_product_attention(Q, K, V)

    print(f"\n输出形状: {output.shape}")
    print(f"权重形状: {weights.shape}")

    print("\n注意力权重矩阵:")
    print("  (每行是一个查询对所有输入的关注程度)")
    for i in range(seq_len_q):
        print(f"  查询 {i}: {weights[i]}")

    # === 2. 加性注意力 ===
    print("\n=== 2. 加性注意力 ===")

    output_add, weights_add = additive_attention(Q, K, V)

    print(f"输出形状: {output_add.shape}")
    print("\n注意力权重矩阵:")
    for i in range(seq_len_q):
        print(f"  查询 {i}: {weights_add[i]}")

    # === 3. 对比两种注意力 ===
    print("\n=== 3.两种注意力对比 ===")

    print("\n点积注意力特点:")
    print("  - 公式: softmax(QK^T / sqrt(d_k)) V")
    print("  - 计算简单：只有矩阵乘法")
    print("  - 效率高：适合大规模")
    print("  -Transformer 使用这种")

    print("\n加性注意力特点:")
    print("  - 公式: softmax(v^T tanh(W_q Q + W_k K)) V")
    print("  - 通过神经网络学习匹配方式")
    print("  - 表达能力更强")
    print("  - 计算较慢")

    # === 4. 掩码的作用 ===
    print("\n=== 4. 掩码示例 ===")

    # 创建因果掩码（上三角为 -inf）
    seq_len = 4
    mask = np.triu(np.ones((seq_len, seq_len)) * -1e9, k=1)

    print("因果掩码矩阵（-1e9表示看不见）:")
    print(mask)

    # 模拟注意力计算
    Q_test = np.random.randn(seq_len, d_k)
    K_test = np.random.randn(seq_len, d_k)
    V_test = np.random.randn(seq_len, d_v)

    _, weights_masked = scaled_dot_product_attention(Q_test, K_test, V_test, mask)

    print("\n应用掩码后的注意力权重:")
    for i in range(seq_len):
        print(f" 位置{i}: {weights_masked[i]}")
    print("  注意：每个位置只能看到自己和前面的位置")

    print("\n" + "=" * 50)
    print("QKV 框架总结:")
    print("Q (Query): 我要找什么")
    print("K (Key):   我是什么类型")
    print("V (Value): 我的具体内容")
    print("Q·K → 相关性 → softmax → 权重 → 加权求和 V")
    print("=" * 50)
