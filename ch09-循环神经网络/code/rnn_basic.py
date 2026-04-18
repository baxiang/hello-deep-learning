"""
第9章 循环神经网络 - 基础 RNN 实现
包含：RNN 单元、完整 RNN 网络、前向传播演示
"""

import numpy as np


class RNNCell:
    """单个 RNN 单元

    核心公式: h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
    """

    def __init__(self, input_size, hidden_size):
        """初始化 RNN 单元

        Args:
            input_size: 输入向量维度
            hidden_size: 隐状态维度
        """
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 初始化权重（使用小随机值防止梯度爆炸）
        self.W_xh = np.random.randn(input_size, hidden_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_h = np.zeros(hidden_size)

        # 保存用于反向传播
        self.x = None
        self.h_prev = None
        self.h = None

    def forward(self, x, h_prev):
        """前向传播

        Args:
            x: 当前输入，形状 (input_size,)
            h_prev: 上一个隐状态，形状 (hidden_size,)

        Returns:
            h: 新的隐状态，形状 (hidden_size,)
        """
        self.x = x
        self.h_prev = h_prev

        # 计算新的隐状态
        self.h = np.tanh(np.dot(x, self.W_xh) + np.dot(h_prev, self.W_hh) + self.b_h)
        return self.h

    def backward(self, dh):
        """反向传播

        Args:
            dh: 对隐状态的梯度，形状 (hidden_size,)

        Returns:
            dx: 对输入的梯度
            dh_prev: 对上一个隐状态的梯度
            grads: 权重梯度字典
        """
        # tanh 的导数: 1 - h^2
        dh_raw = dh * (1 - self.h**2)

        # 计算各梯度
        dx = np.dot(dh_raw, self.W_xh.T)
        dh_prev = np.dot(dh_raw, self.W_hh.T)

        dW_xh = np.outer(self.x, dh_raw)
        dW_hh = np.outer(self.h_prev, dh_raw)
        db_h = dh_raw

        grads = {"dW_xh": dW_xh, "dW_hh": dW_hh, "db_h": db_h}

        return dx, dh_prev, grads

    def __call__(self, x, h_prev):
        return self.forward(x, h_prev)


class SimpleRNN:
    """完整的 RNN 网络"""

    def __init__(self, input_size, hidden_size, output_size=None):
        """初始化 RNN 网络

        Args:
            input_size: 输入向量维度
            hidden_size: 隐状态维度
            output_size: 输出向量维度（可选，若为 None 则只返回隐状态）
        """
        self.hidden_size = hidden_size
        self.rnn_cell = RNNCell(input_size, hidden_size)
        self.output_size = output_size

        if output_size is not None:
            # 输出层权重
            self.W_hy = np.random.randn(hidden_size, output_size) * 0.01
            self.b_y = np.zeros(output_size)

        # 保存前向传播结果用于反向传播
        self.xs = None
        self.hs = None

    def forward(self, x_seq, h0=None):
        """处理整个序列

        Args:
            x_seq: 输入序列，形状 (seq_len, input_size)
            h0: 初始隐状态（可选，默认为零向量）

        Returns:
            outputs: 所有时间步的输出（若有输出层）
            hidden_states: 所有隐状态
            h_final: 最终隐状态
        """
        seq_len = x_seq.shape[0]

        # 初始化隐状态
        if h0 is None:
            h = np.zeros(self.hidden_size)
        else:
            h = h0.copy()

        # 保存输入和隐状态
        self.xs = []
        self.hs = [h.copy()]

        outputs = []

        for t in range(seq_len):
            h = self.rnn_cell(x_seq[t], h)
            self.xs.append(x_seq[t].copy())
            self.hs.append(h.copy())

            if self.output_size is not None:
                y = np.dot(h, self.W_hy) + self.b_y
                outputs.append(y.copy())

        h_final = h

        if self.output_size is not None:
            return np.array(outputs), np.array(self.hs[1:]), h_final
        else:
            return np.array(self.hs[1:]), h_final

    def __call__(self, x_seq, h0=None):
        return self.forward(x_seq, h0)


def softmax(x):
    """Softmax 函数"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


if __name__ == "__main__":
    print("=" * 50)
    print("RNN 基础实现演示")
    print("=" * 50)

    np.random.seed(42)

    # === 1. 单个 RNN 单元测试 ===
    print("\n=== 1. RNN 单元测试 ===")
    cell = RNNCell(input_size=4, hidden_size=8)

    x = np.random.randn(4)
    h_prev = np.zeros(8)

    h = cell.forward(x, h_prev)
    print(f"输入形状: {x.shape}")
    print(f"隐状态形状: {h.shape}")
    print(f"隐状态范数: {np.linalg.norm(h):.4f}")

    # === 2. 完整 RNN 网络测试 ===
    print("\n=== 2. 完整 RNN 网络测试 ===")
    rnn = SimpleRNN(input_size=4, hidden_size=8, output_size=3)

    # 模拟输入序列：5 个时间步
    seq_len = 5
    x_seq = np.random.randn(seq_len, 4)

    outputs, hidden_states, h_final = rnn(x_seq)

    print(f"输入序列形状: {x_seq.shape}")
    print(f"隐状态序列形状: {hidden_states.shape}")
    print(f"输出序列形状: {outputs.shape}")

    print("\n各时间步隐状态范数:")
    for t in range(seq_len):
        print(f"  t={t}: {np.linalg.norm(hidden_states[t]):.4f}")

    # === 3. 多对一任务示例（序列分类） ===
    print("\n=== 3. 多对一任务示例 ===")
    # 只取最后一个时间步的输出
    last_output = outputs[-1]
    last_probs = softmax(last_output)
    print("最后时间步的输出概率:", last_probs)
    print("预测类别:", np.argmax(last_probs))

    # === 4. 观察记忆传递 ===
    print("\n=== 4. 记忆传递演示 ===")
    # 创建两个不同的输入序列
    x_seq1 = np.random.randn(seq_len, 4)
    x_seq2 = np.random.randn(seq_len, 4)

    _, hs1, _ = rnn(x_seq1)
    _, hs2, _ = rnn(x_seq2)

    # 计算两个序列最终隐状态的差异
    diff = np.linalg.norm(hs1[-1] - hs2[-1])
    print(f"两个不同序列最终隐状态的差异: {diff:.4f}")
    print("这说明不同的输入序列会产生不同的'记忆'")

    print("\n" + "=" * 50)
    print("RNN 关键点总结:")
    print("1. 隐状态 h 是 RNN 的记忆，会随时间传递")
    print("2. tanh 激活函数让隐状态保持在 [-1, 1] 范围")
    print("3. 每个时间步的输出依赖当前输入和历史记忆")
    print("=" * 50)
