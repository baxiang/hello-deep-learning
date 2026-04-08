import numpy as np


def relu(x):
    return np.maximum(0, x)


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    return exp_a / np.sum(exp_a)


class ResidualBlock:
    """残差块"""

    def __init__(self, input_size, hidden_size):
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(input_size)

    def forward(self, x):
        h = relu(np.dot(x, self.W1) + self.b1)
        out = np.dot(h, self.W2) + self.b2
        return out + x


class DeepNetwork:
    """深层残差网络"""

    def __init__(self, input_size, hidden_size, num_blocks, output_size):
        self.blocks = []
        for _ in range(num_blocks):
            self.blocks.append(ResidualBlock(input_size, hidden_size))
        self.W_out = np.random.randn(input_size, output_size) * 0.01
        self.b_out = np.zeros(output_size)

    def predict(self, x):
        for block in self.blocks:
            x = block.forward(x)
        return softmax(np.dot(x, self.W_out) + self.b_out)


def search_hyperparams(num_trials=10):
    """超参数随机搜索"""
    best_score = 0
    best_params = {}

    for trial in range(num_trials):
        lr = 10 ** np.random.uniform(-6, -2)
        weight_decay = 10 ** np.random.uniform(-8, -4)
        batch_size = int(2 ** np.random.uniform(4, 8))
        dropout = np.random.uniform(0, 0.5)

        # 模拟评估分数
        score = 0.5 + np.random.normal(0, 0.05)
        if 0.001 <= lr <= 0.01:
            score += 0.1
        if 1e-6 <= weight_decay <= 1e-4:
            score += 0.05
        if 32 <= batch_size <= 128:
            score += 0.05
        if 0.2 <= dropout <= 0.4:
            score += 0.05
        score = min(score, 0.99)

        if score > best_score:
            best_score = score
            best_params = {
                "lr": lr,
                "wd": weight_decay,
                "batch": batch_size,
                "dropout": dropout,
            }

    return best_score, best_params


if __name__ == "__main__":
    print("=" * 50)
    print("深度学习全书总结 - 演示代码")
    print("=" * 50)

    # 1. 深层残差网络
    print("\n[1] 深层残差网络")
    np.random.seed(42)
    net = DeepNetwork(input_size=64, hidden_size=32, num_blocks=5, output_size=10)
    x = np.random.randn(1, 64)
    y = net.predict(x)
    print(f"  输入: {x.shape} → 输出: {y.shape}")
    print(f"  预测: {np.argmax(y[0])}, 概率: {np.max(y[0]):.4f}")
    print(f"  输出总和: {np.sum(y[0]):.6f}")

    # 2. 超参数搜索
    print("\n[2] 超参数随机搜索 (10 次)")
    np.random.seed(42)
    best_score, best_params = search_hyperparams(10)
    print(f"  最佳准确率: {best_score:.4f}")
    print(f"  最佳参数: {best_params}")

    print("\n" + "=" * 50)
    print("恭喜完成全书学习！")
    print("=" * 50)
