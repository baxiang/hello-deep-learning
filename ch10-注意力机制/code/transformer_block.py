"""
第10章 注意力机制 - Transformer 块简化版
包含：多头注意力、残差连接、层归一化、前馈网络
"""

import numpy as np


def softmax(x, axis=-1):
    """Softmax 函数"""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def layer_norm(x, eps=1e-6):
    """层归一化

    对每个样本的所有特征进行归一化

    Args:
        x: 输入，形状 (..., d)
        eps: 防止除零的小值

    Returns:
        归一化后的值
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + eps)


class MultiHeadAttention:
    """多头注意力模块"""

    def __init__(self, d_model, num_heads):
        """初始化

        Args:
            d_model: 模型维度
            num_heads: 头的数量
        """
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # QKV 权重
        np.random.seed(42)
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        self.W_o = np.random.randn(d_model, d_model) * 0.1

    def forward(self, X, mask=None):
        """多头注意力前向传播

        Args:
            X: 输入，形状 (seq_len, d_model)
            mask: 可选掩码

        Returns:
            output: 输出
            all_weights: 各头的注意力权重
        """
        seq_len = X.shape[0]

        # 计算 Q、K、V
        Q = np.dot(X, self.W_q)
        K = np.dot(X, self.W_k)
        V = np.dot(X, self.W_v)

        # 分头reshape 成 (num_heads, seq_len, d_k)
        Q_heads = Q.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)
        K_heads = K.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)
        V_heads = V.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)

        # 各头计算注意力
        all_weights = []
        all_outputs = []

        for h in range(self.num_heads):
            # 点积注意力
            scores = np.dot(Q_heads[h], K_heads[h].T) / np.sqrt(self.d_k)

            if mask is not None:
                scores = scores + mask

            weights = softmax(scores, axis=-1)
            all_weights.append(weights)

            head_output = np.dot(weights, V_heads[h])
            all_outputs.append(head_output)

        # 拼接所有头
        concat = np.concatenate(all_outputs, axis=-1)

        # 输出投影
        output = np.dot(concat, self.W_o)

        return output, np.array(all_weights)


class FeedForward:
    """前馈网络"""

    def __init__(self, d_model, d_ff):
        """初始化

        Args:
            d_model: 输入输出维度
            d_ff: 隐藏层维度
        """
        np.random.seed(42)
        self.W_1 = np.random.randn(d_model, d_ff) * 0.1
        self.b_1 = np.zeros(d_ff)
        self.W_2 = np.random.randn(d_ff, d_model) * 0.1
        self.b_2 = np.zeros(d_model)

    def forward(self, X):
        """前向传播

        FFN(x) = max(0, xW_1 + b_1) W_2 + b_2

        Args:
            X: 输入，形状 (seq_len, d_model)

        Returns:
            输出，形状 (seq_len, d_model)
        """
        hidden = np.maximum(0, np.dot(X, self.W_1) + self.b_1)  # ReLU
        output = np.dot(hidden, self.W_2) + self.b_2
        return output


class TransformerBlock:
    """Transformer 块"""

    def __init__(self, d_model, num_heads, d_ff):
        """初始化

        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
        """
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, X, mask=None):
        """Transformer 块前向传播

        结构:
            X → MultiHeadAttention → Add & Norm → FFN → Add & Norm → Output

        Args:
            X: 输入
            mask: 可选掩码

        Returns:
            output: 输出
            weights: 注意力权重
        """
        # 多头注意力 + 残差连接 + 层归一化
        attn_out, weights = self.mha.forward(X, mask)
        X = layer_norm(X + attn_out)  # Add & Norm

        # 前馈网络 + 残差连接 + 层归一化
        ffn_out = self.ffn.forward(X)
        X = layer_norm(X + ffn_out)  # Add & Norm

        return X, weights


class SimpleTransformer:
    """简化版 Transformer（Encoder 部分）"""

    def __init__(self, d_model=32, num_heads=4, d_ff=128, num_layers=2):
        """初始化

        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络维度
            num_layers:层数
        """
        self.d_model = d_model
        self.num_layers = num_layers

        self.layers = [
            TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)
        ]

    def forward(self, X, mask=None):
        """前向传播

        Args:
            X: 输入，形状 (seq_len, d_model)
            mask: 可选掩码

        Returns:
            output: 最终输出
            all_weights: 各层的注意力权重
        """
        all_weights = []

        for layer in self.layers:
            X, weights = layer.forward(X, mask)
            all_weights.append(weights)

        return X, all_weights


def positional_encoding(seq_len, d_model):
    """位置编码"""
    PE = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(d_model):
            if i % 2 == 0:
                PE[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            else:
                PE[pos, i] = np.cos(pos / (10000 ** ((i - 1) / d_model)))
    return PE


if __name__ == "__main__":
    print("=" * 50)
    print("Transformer 块演示")
    print("=" * 50)

    np.random.seed(42)

    # 配置
    d_model = 16
    num_heads = 4
    d_ff = 32
    num_layers = 2

    print("\n配置:")
    print(f"  模型维度: {d_model}")
    print(f"  注意力头数: {num_heads}")
    print(f"  前馈网络维度: {d_ff}")
    print(f"  层数: {num_layers}")

    # === 1. 单个 Transformer 块 ===
    print("\n=== 1. 单个 Transformer 块 ===")

    seq_len = 4
    X = np.random.randn(seq_len, d_model)

    block = TransformerBlock(d_model, num_heads, d_ff)
    output, weights = block.forward(X)

    print(f"输入形状: {X.shape}")
    print(f"输出形状: {output.shape}")

    print("\n各头注意力权重:")
    words = ["词1", "词2", "词3", "词4"]
    for h in range(num_heads):
        print(f"\n头 {h}:")
        for i in range(seq_len):
            print(f"  {words[i]} 关注: ", end="")
            for j in range(seq_len):
                print(f"{words[j]}({weights[h][i, j]:.2f}) ", end="")
            print()

    # === 2. 多层 Transformer ===
    print("\n=== 2. 多层 Transformer ===")

    transformer = SimpleTransformer(d_model, num_heads, d_ff, num_layers)

    # 加入位置编码
    PE = positional_encoding(seq_len, d_model)
    X_with_pos = X + PE

    final_output, all_weights = transformer.forward(X_with_pos)

    print(f"最终输出形状: {final_output.shape}")
    print(f"层数: {len(all_weights)}")

    print("\n各层的注意力模式变化:")
    for layer_idx, layer_weights in enumerate(all_weights):
        # 找每层最关注的模式
        avg_weights = np.mean(layer_weights, axis=0)  # 平均各头
        max_focus = np.argmax(avg_weights, axis=1)
        print(f"  层 {layer_idx}: 平均最大关注")
        for i in range(seq_len):
            print(f"    {words[i]} → {words[max_focus[i]]}")

    # === 3. 残差连接和 LayerNorm 的作用 ===
    print("\n=== 3. 残差连接 & LayerNorm ===")

    # 对比有无残差连接
    X_test = np.random.randn(seq_len, d_model)

    # 有残差连接
    attn_out_test, _ = block.mha.forward(X_test)
    with_residual = X_test + attn_out_test

    # 无残差连接
    without_residual = attn_out_test

    print("输出范数对比:")
    print(f"  有残差: {np.linalg.norm(with_residual):.4f}")
    print(f"  无残差: {np.linalg.norm(with_residual):.4f}")

    print("\n残差连接的作用:")
    print("  - 让梯度更容易传播")
    print("  - 防止深层网络的梯度消失")
    print("  - 每层只需学习'增量'")

    print("\nLayerNorm 的作用:")
    print("  - 稳定每层输入的分布")
    print("  - 加速训练收敛")
    print("  - 对每个样本独立归一化")

    print("\n" + "=" * 50)
    print("Transformer 块总结:")
    print("1. 多头注意力：多角度看信息")
    print("2. 残差连接：输入 + 子层输出")
    print("3. LayerNorm：稳定训练")
    print("4. 前馈网络：增加非线性变换")
    print("5. 多层堆叠：深度建模")
    print("=" * 50)
