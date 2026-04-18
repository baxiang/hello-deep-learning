"""
第9章 循环神经网络 - 文本生成示例
包含：字符级语言模型、训练、文本生成
"""

import numpy as np


def softmax(x):
    """Softmax 函数"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class CharRNN:
    """字符级 RNN 语言模型

    用于学习文本的字符序列规律，并生成新文本
    """

    def __init__(self, vocab_size, hidden_size, learning_rate=0.1):
        """初始化模型

        Args:
            vocab_size: 词汇表大小（字符种类数）
            hidden_size: 隐状态维度
            learning_rate: 学习率
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # 初始化权重
        self.W_xh = np.random.randn(vocab_size, hidden_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(hidden_size, vocab_size) * 0.01
        self.b_h = np.zeros(hidden_size)
        self.b_y = np.zeros(vocab_size)

        # 用于 Adagrad 优化器
        self.h_cache = None

    def forward(self, inputs, h_prev):
        """前向传播

        Args:
            inputs: 输入字符索引列表
            h_prev: 初始隐状态

        Returns:
            xs: 所有 one-hot 输入
            hs: 所有隐状态
            ps: 所有输出概率
            h: 最终隐状态
        """
        xs, hs, ps = {}, {}, {}
        hs[-1] = h_prev.copy()

        for t, idx in enumerate(inputs):
            # One-hot 编码
            xs[t] = np.zeros(self.vocab_size)
            xs[t][idx] = 1

            # RNN 前向传播
            hs[t] = np.tanh(
                np.dot(xs[t], self.W_xh) + np.dot(hs[t - 1], self.W_hh) + self.b_h
            )

            # 输出层
            ys = np.dot(hs[t], self.W_hy) + self.b_y
            ps[t] = softmax(ys)

        return xs, hs, ps, hs[len(inputs) - 1]

    def backward(self, inputs, targets, xs, hs, ps):
        """反向传播

        Args:
            inputs: 输入字符索引列表
            targets: 目标字符索引列表
            xs, hs, ps: 前向传播的结果

        Returns:
            grads: 各参数的梯度
        """
        # 初始化梯度
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)
        dh_next = np.zeros(self.hidden_size)

        # 反向传播
        for t in reversed(range(len(inputs))):
            # 输出层梯度（交叉熵）
            dy = ps[t].copy()
            dy[targets[t]] -= 1

            dW_hy += np.outer(hs[t], dy)
            db_y += dy

            # 隐状态梯度
            dh = np.dot(dy, self.W_hy.T) + dh_next
            dh_raw = (1 - hs[t] ** 2) * dh  # tanh 导数

            dW_xh += np.outer(xs[t], dh_raw)
            dW_hh += np.outer(hs[t - 1], dh_raw)
            db_h += dh_raw

            dh_next = np.dot(dh_raw, self.W_hh.T)

        # 梯度裁剪
        grads = [dW_xh, dW_hh, dW_hy, db_h, db_y]
        for grad in grads:
            np.clip(grad, -5, 5, out=grad)

        return grads

    def update_adagrad(self, grads):
        """使用 Adagrad 更新参数"""
        dW_xh, dW_hh, dW_hy, db_h, db_y = grads

        if self.h_cache is None:
            self.h_cache = [
                np.zeros_like(p)
                for p in [self.W_xh, self.W_hh, self.W_hy, self.b_h, self.b_y]
            ]

        params = [self.W_xh, self.W_hh, self.W_hy, self.b_h, self.b_y]
        for i, (param, grad, h) in enumerate(zip(params, grads, self.h_cache)):
            h += grad**2
            param -= self.learning_rate * grad / (np.sqrt(h) + 1e-8)
            self.h_cache[i] = h

    def loss(self, ps, targets):
        """计算交叉熵损失"""
        return sum(-np.log(ps[t][targets[t]] + 1e-10) for t in range(len(targets)))

    def sample(self, seed_idx, h, length, temperature=1.0):
        """生成文本

        Args:
            seed_idx: 种子字符索引
            h: 初始隐状态
            length: 生成长度
            temperature: 温度参数（控制随机性）

        Returns:
            生成的字符索引列表
        """
        indices = [seed_idx]
        x = np.zeros(self.vocab_size)
        x[seed_idx] = 1

        for _ in range(length):
            # 前向传播
            h = np.tanh(np.dot(x, self.W_xh) + np.dot(h, self.W_hh) + self.b_h)
            y = np.dot(h, self.W_hy) + self.b_y

            # 应用温度
            y = y / temperature
            p = softmax(y)

            # 采样
            idx = np.random.choice(self.vocab_size, p=p)
            indices.append(idx)

            # 准备下一个输入
            x = np.zeros(self.vocab_size)
            x[idx] = 1

        return indices


def prepare_data(text):
    """准备训练数据

    Args:
        text: 原始文本

    Returns:
        char_to_idx: 字符到索引的映射
        idx_to_char: 索引到字符的映射
        vocab_size: 词汇表大小
    """
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    vocab_size = len(chars)

    return char_to_idx, idx_to_char, vocab_size


def train(model, text, char_to_idx, seq_length, epochs, print_every=100):
    """训练模型

    Args:
        model: CharRNN 模型
        text: 训练文本
        char_to_idx: 字符索引映射
        seq_length: 序列长度
        epochs: 训练轮数
        print_every: 打印间隔

    Returns:
        losses: 损失历史
    """
    losses = []
    h = np.zeros(model.hidden_size)

    for epoch in range(epochs):
        # 随机选择起始位置
        start = np.random.randint(0, len(text) - seq_length - 1)
        inputs = [char_to_idx[c] for c in text[start : start + seq_length]]
        targets = [char_to_idx[c] for c in text[start + 1 : start + seq_length + 1]]

        # 前向传播
        xs, hs, ps, h = model.forward(inputs, h)

        # 计算损失
        loss = model.loss(ps, targets)
        losses.append(loss)

        # 反向传播
        grads = model.backward(inputs, targets, xs, hs, ps)

        # 更新参数
        model.update_adagrad(grads)

        # 每 N 步重置隐状态
        if epoch % seq_length == 0:
            h = np.zeros(model.hidden_size)

        if epoch % print_every == 0:
            print(f"Epoch {epoch:4d}, Loss: {loss:.4f}")

    return losses


def generate(model, seed_char, char_to_idx, idx_to_char, length=50, temperature=1.0):
    """生成文本

    Args:
        model: 训练好的模型
        seed_char: 种子字符
        char_to_idx, idx_to_char: 字符映射
        length: 生成长度
        temperature: 温度参数

    Returns:
        生成的文本
    """
    seed_idx = char_to_idx[seed_char]
    h = np.zeros(model.hidden_size)

    indices = model.sample(seed_idx, h, length, temperature)
    text = "".join(idx_to_char[i] for i in indices)

    return text


if __name__ == "__main__":
    print("=" * 50)
    print("字符级 RNN 文本生成演示")
    print("=" * 50)

    np.random.seed(42)

    # 训练文本（简单的中文句子）
    text = "我喜欢深度学习，深度学习很有趣。神经网络很强大，神经网络能学习。学习深度学习需要耐心，深度学习很有用。"

    print(f"\n训练文本: {text}")
    print(f"文本长度: {len(text)} 字符")

    # 准备数据
    char_to_idx, idx_to_char, vocab_size = prepare_data(text)
    print(f"词汇表大小: {vocab_size}")
    print(f"字符表: {list(char_to_idx.keys())}")

    # 创建模型
    hidden_size = 32
    learning_rate = 0.1
    model = CharRNN(vocab_size, hidden_size, learning_rate)

    print(f"\n模型配置: hidden_size={hidden_size}, lr={learning_rate}")

    # 训练
    print("\n=== 开始训练 ===")
    seq_length = 15
    epochs = 1000
    losses = train(model, text, char_to_idx, seq_length, epochs, print_every=200)

    # 生成文本
    print("\n=== 文本生成 ===")
    seed_char = "我"

    for temp in [0.3, 0.7, 1.0, 1.5]:
        generated = generate(
            model, seed_char, char_to_idx, idx_to_char, length=30, temperature=temp
        )
        print(f"温度 {temp:.1f}: {generated}")

    print("\n" + "=" * 50)
    print("文本生成关键点:")
    print("1. 字符级模型：每个字符是一个 token")
    print("2. 训练目标：预测下一个字符")
    print("3. 生成方式：采样 + 温度控制")
    print("4. 温度低 = 更确定，温度高 = 更随机")
    print("=" * 50)
