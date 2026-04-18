"""
第9章 循环神经网络 - LSTM 单元实现
包含：LSTM 单元、完整 LSTM 网络、三个门控机制演示
"""

import numpy as np


class LSTMCell:
    """LSTM 单元实现

    LSTM 有三个门控：
    - 遗忘门 (f): 决定丢弃哪些旧信息
    - 输入门 (i): 决定存入哪些新信息
    - 输出门 (o): 决定输出哪些信息

    还有一个细胞状态 C，用于存储长期记忆
    """

    def __init__(self, input_size, hidden_size):
        """初始化 LSTM 单元

        Args:
            input_size: 输入向量维度
            hidden_size: 隐状态维度（也是细胞状态维度）
        """
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 合并所有门的权重，提高计算效率
        # 顺序: [遗忘门, 输入门, 候选记忆, 输出门]
        self.W_x = np.random.randn(input_size, hidden_size * 4) * 0.01
        self.W_h = np.random.randn(hidden_size, hidden_size * 4) * 0.01
        self.b = np.zeros(hidden_size * 4)

        # 分离各个门（用于演示）
        self._split_gates()

        # 保存前向传播结果
        self.cache = None

    def _split_gates(self):
        """分离各个门的权重"""
        h = self.hidden_size
        self.W_xf, self.W_xi, self.W_xc, self.W_xo = [
            self.W_x[:, h * i : h * (i + 1)] for i in range(4)
        ]
        self.W_hf, self.W_hi, self.W_hc, self.W_ho = [
            self.W_h[:, h * i : h * (i + 1)] for i in range(4)
        ]
        self.b_f, self.b_i, self.b_c, self.b_o = [
            self.b[h * i : h * (i + 1)] for i in range(4)
        ]

    def sigmoid(self, x):
        """Sigmoid 函数"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, x, h_prev, c_prev):
        """LSTM 前向传播

        Args:
            x: 当前输入，形状 (input_size,)
            h_prev: 上一个隐状态，形状 (hidden_size,)
            c_prev: 上一个细胞状态，形状 (hidden_size,)

        Returns:
            h: 新的隐状态
            c: 新的细胞状态
        """
        # 合并计算所有门（提高效率）
        gates = np.dot(x, self.W_x) + np.dot(h_prev, self.W_h) + self.b

        # 分离各个门
        f = self.sigmoid(gates[: self.hidden_size])  # 遗忘门
        i = self.sigmoid(gates[self.hidden_size : 2 * self.hidden_size])  # 输入门
        c_tilde = np.tanh(
            gates[2 * self.hidden_size : 3 * self.hidden_size]
        )  # 候选记忆
        o = self.sigmoid(gates[3 * self.hidden_size :])  # 输出门

        # 更新细胞状态：遗忘旧记忆 + 加入新记忆
        c = f * c_prev + i * c_tilde

        # 计算隐状态
        h = o * np.tanh(c)

        # 保存用于反向传播
        self.cache = (x, h_prev, c_prev, f, i, c_tilde, o, c)

        return h, c

    def backward(self, dh, dc):
        """LSTM 反向传播（简化版）

        Args:
            dh: 对隐状态的梯度
            dc: 对细胞状态的梯度

        Returns:
            dx: 对输入的梯度
            dh_prev: 对上一个隐状态的梯度
            dc_prev: 对上一个细胞状态的梯度
        """
        x, h_prev, c_prev, f, i, c_tilde, o, c = self.cache

        # 输出门的梯度
        do = dh * np.tanh(c)
        dc_from_h = dh * o * (1 - np.tanh(c) ** 2)
        dc_total = dc + dc_from_h

        # 细胞状态的梯度
        df = dc_total * c_prev
        dc_prev = dc_total * f
        di = dc_total * c_tilde
        dc_tilde = dc_total * i

        # sigmoid 和 tanh 的导数
        df_raw = df * f * (1 - f)
        di_raw = di * i * (1 - i)
        dc_tilde_raw = dc_tilde * (1 - c_tilde**2)
        do_raw = do * o * (1 - o)

        # 合并门梯度
        d_gates = np.concatenate([df_raw, di_raw, dc_tilde_raw, do_raw])

        # 计算对输入和上一状态的梯度
        dx = np.dot(d_gates, self.W_x.T)
        dh_prev = np.dot(d_gates, self.W_h.T)

        return dx, dh_prev, dc_prev

    def __call__(self, x, h_prev, c_prev):
        return self.forward(x, h_prev, c_prev)


class LSTM:
    """完整的 LSTM 网络"""

    def __init__(self, input_size, hidden_size, output_size=None):
        """初始化 LSTM 网络"""
        self.hidden_size = hidden_size
        self.lstm_cell = LSTMCell(input_size, hidden_size)
        self.output_size = output_size

        if output_size is not None:
            self.W_hy = np.random.randn(hidden_size, output_size) * 0.01
            self.b_y = np.zeros(output_size)

        # 保存前向传播结果
        self.hs = None
        self.cs = None

    def forward(self, x_seq, h0=None, c0=None):
        """处理整个序列

        Args:
            x_seq: 输入序列，形状 (seq_len, input_size)
            h0: 初始隐状态（可选）
            c0: 初始细胞状态（可选）

        Returns:
            outputs: 所有输出（若有输出层）
            hidden_states: 所有隐状态
            cell_states: 所有细胞状态
        """
        seq_len = x_seq.shape[0]

        # 初始化
        h = h0 if h0 is not None else np.zeros(self.hidden_size)
        c = c0 if c0 is not None else np.zeros(self.hidden_size)

        self.hs = [h.copy()]
        self.cs = [c.copy()]

        outputs = []

        for t in range(seq_len):
            h, c = self.lstm_cell(x_seq[t], h, c)
            self.hs.append(h.copy())
            self.cs.append(c.copy())

            if self.output_size is not None:
                y = np.dot(h, self.W_hy) + self.b_y
                outputs.append(y.copy())

        if self.output_size is not None:
            return np.array(outputs), np.array(self.hs[1:]), np.array(self.cs[1:])
        else:
            return np.array(self.hs[1:]), np.array(self.cs[1:])

    def __call__(self, x_seq, h0=None, c0=None):
        return self.forward(x_seq, h0, c0)


if __name__ == "__main__":
    print("=" * 50)
    print("LSTM 单元实现演示")
    print("=" * 50)

    np.random.seed(42)

    # === 1. LSTM 单元测试 ===
    print("\n=== 1. LSTM 单元测试 ===")
    cell = LSTMCell(input_size=4, hidden_size=8)

    x = np.random.randn(4)
    h_prev = np.zeros(8)
    c_prev = np.zeros(8)

    h, c = cell.forward(x, h_prev, c_prev)
    print(f"输入形状: {x.shape}")
    print(f"隐状态形状: {h.shape}")
    print(f"细胞状态形状: {c.shape}")
    print(f"隐状态范数: {np.linalg.norm(h):.4f}")
    print(f"细胞状态范数: {np.linalg.norm(c):.4f}")

    # === 2. 观察门控值 ===
    print("\n=== 2. 观察门控值 ===")
    # 重新运行以获取门控值
    gates = np.dot(x, cell.W_x) + np.dot(h_prev, cell.W_h) + cell.b
    f = cell.sigmoid(gates[:8])
    i = cell.sigmoid(gates[8:16])
    o = cell.sigmoid(gates[24:])

    print(f"遗忘门 f 平均值: {f.mean():.4f} (接近1=保留，接近0=遗忘)")
    print(f"输入门 i 平均值: {i.mean():.4f} (接近1=存入新信息)")
    print(f"输出门 o 平均值: {o.mean():.4f} (接近1=输出)")

    # === 3. 完整 LSTM 网络测试 ===
    print("\n=== 3. 完整 LSTM 网络测试 ===")
    lstm = LSTM(input_size=4, hidden_size=8, output_size=3)

    seq_len = 5
    x_seq = np.random.randn(seq_len, 4)

    outputs, hs, cs = lstm(x_seq)

    print(f"输入序列形状: {x_seq.shape}")
    print(f"隐状态序列形状: {hs.shape}")
    print(f"细胞状态序列形状: {cs.shape}")

    print("\n细胞状态变化（长期记忆）:")
    for t in range(seq_len):
        print(f"  t={t}: 范数 {np.linalg.norm(cs[t]):.4f}")

    print("\n隐状态变化（短期记忆+输出）:")
    for t in range(seq_len):
        print(f"  t={t}: 范数 {np.linalg.norm(hs[t]):.4f}")

    # === 4. LSTM vs RNN 记忆对比 ===
    print("\n=== 4. LSTM vs RNN 记忆对比 ===")
    # 模拟长序列，看谁能保持记忆
    long_seq_len = 20
    x_long = np.random.randn(long_seq_len, 4)

    _, hs_lstm, cs_lstm = LSTM(input_size=4, hidden_size=8)(x_long)

    # RNN 的隐状态（假设用 tanh）
    h_rnn = np.zeros(8)
    hs_rnn = []
    W_xh = np.random.randn(4, 8) * 0.01
    W_hh = np.random.randn(8, 8) * 0.01
    for t in range(long_seq_len):
        h_rnn = np.tanh(np.dot(x_long[t], W_xh) + np.dot(h_rnn, W_hh))
        hs_rnn.append(h_rnn.copy())

    print("隐状态稳定性（方差）:")
    print(f"  LSTM: {np.var([np.linalg.norm(h) for h in hs_lstm]):.4f}")
    print(f"  RNN:  {np.var([np.linalg.norm(h) for h in hs_rnn]):.4f}")
    print("LSTM 的细胞状态提供了更稳定的记忆传递")

    print("\n" + "=" * 50)
    print("LSTM 关键点总结:")
    print("1. 三个门控精确控制信息流动")
    print("2. 细胞状态 C 是长期记忆的'高速公路'")
    print("3. 隐状态 h 是短期记忆和输出")
    print("4. 解决了 RNN 的长距离依赖问题")
    print("=" * 50)
