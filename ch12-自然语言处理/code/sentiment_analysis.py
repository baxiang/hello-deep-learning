"""
第12章 自然语言处理 - 情感分析简化实现
包含：文本预处理、简单分类模型
"""

import numpy as np


def softmax(x):
    """Softmax 函数"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class TextProcessor:
    """文本预处理工具"""

    def __init__(self, max_len=20):
        """初始化

        Args:
            max_len: 最大序列长度
        """
        self.max_len = max_len
        self.vocab = {}
        self.pad_idx = 0
        self.unk_idx = 1

    def tokenize(self, text):
        """分词（简化：假设已分词）"""
        return text.split()

    def build_vocab(self, texts, min_freq=1):
        """构建词表"""
        word_counts = {}

        for text in texts:
            words = self.tokenize(text)
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

        # 构建词表
        self.vocab = {"[PAD]": 0, "[UNK]": 1}
        idx = 2

        for word, count in word_counts.items():
            if count >= min_freq:
                self.vocab[word] = idx
                idx += 1

        return self.vocab

    def text_to_indices(self, text):
        """文本转索引序列"""
        words = self.tokenize(text)
        indices = []

        for word in words:
            if word in self.vocab:
                indices.append(self.vocab[word])
            else:
                indices.append(self.unk_idx)

        return indices

    def pad_sequence(self, indices):
        """填充/截断到固定长度"""
        if len(indices) < self.max_len:
            indices = indices + [self.pad_idx] * (self.max_len - len(indices))
        else:
            indices = indices[: self.max_len]

        return indices

    def process(self, text):
        """完整预处理"""
        indices = self.text_to_indices(text)
        padded = self.pad_sequence(indices)
        return np.array(padded)


class SimpleTextClassifier:
    """简单的文本分类模型"""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        """初始化

        Args:
            vocab_size: 词表大小
            embedding_dim: 词向量维度
            hidden_dim: 隐藏层维度
            num_classes: 类别数
        """
        np.random.seed(42)

        # 词嵌入层
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.1

        # RNN 权重（简化版）
        self.W_h = np.random.randn(embedding_dim, hidden_dim) * 0.1
        self.b_h = np.zeros(hidden_dim)

        # 分类层
        self.W_out = np.random.randn(hidden_dim, num_classes) * 0.1
        self.b_out = np.zeros(num_classes)

    def forward(self, x_indices):
        """前向传播

        Args:
            x_indices: 词索引序列，形状 (seq_len,)

        Returns:
            probs: 分类概率
            hidden: 最终隐状态
        """
        # 词嵌入
        embedded = self.embeddings[x_indices]  # (seq_len, embedding_dim)

        # 简化 RNN：用均值作为特征
        # 实际应用用 LSTM/GRU
        hidden = np.tanh(np.mean(embedded, axis=0))  # (embedding_dim,)

        # 分类
        logits = np.dot(hidden, self.W_out) + self.b_out
        probs = softmax(logits)

        return probs, hidden


def train_model(model, train_data, epochs=100, lr=0.1):
    """训练模型（简化版）"""
    losses = []

    for epoch in range(epochs):
        total_loss = 0

        for x_indices, y_label in train_data:
            # 前向传播
            probs, hidden = model.forward(x_indices)

            # 计算损失（交叉熵）
            loss = -np.log(probs[y_label] + 1e-10)
            total_loss += loss

            # 简化反向传播
            # 梯度：d_loss/d_logits = probs - target
            d_logits = probs.copy()
            d_logits[y_label] -= 1

            # 更新输出层
            model.W_out -= lr * np.outer(hidden, d_logits)
            model.b_out -= lr * d_logits

        losses.append(total_loss)

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

    return losses


def evaluate_model(model, test_data):
    """评估模型"""
    correct = 0
    total = len(test_data)

    for x_indices, y_label in test_data:
        probs, _ = model.forward(x_indices)
        predicted = np.argmax(probs)

        if predicted == y_label:
            correct += 1

    accuracy = correct / total
    return accuracy


if __name__ == "__main__":
    print("=" * 50)
    print("情感分析演示")
    print("=" * 50)

    # === 1. 数据准备 ===
    print("\n=== 1. 数据准备 ===")

    # 示例数据（已分词）
    train_texts = [
        "这电影 很 精彩",  # 褒义
        "服务 很好",  # 褒义
        "太差了",  # 贬义
        "很失望",  # 贬义
        "还不错",  # 中性
        "一般般",  # 中性
    ]

    train_labels = [0, 0, 1, 1, 2, 2]  # 0=褒义, 1=贬义, 2=中性

    print(f"训练数据: {len(train_texts)} 条")
    print("类别: 褒义(0), 贬义(1), 中性(2)")

    # === 2. 预处理 ===
    print("\n=== 2. 预处理 ===")

    processor = TextProcessor(max_len=10)
    vocab = processor.build_vocab(train_texts)

    print(f"词表大小: {len(vocab)}")
    print(f"词表: {vocab}")

    # 处理数据
    train_data = []
    for text, label in zip(train_texts, train_labels):
        indices = processor.process(text)
        train_data.append((indices, label))

    print("\n预处理示例:")
    print(f"原文: '{train_texts[0]}'")
    print(f"索引: {train_data[0][0]}")

    # === 3. 模型训练 ===
    print("\n=== 3. 模型训练 ===")

    vocab_size = len(vocab)
    embedding_dim = 8
    hidden_dim = 8
    num_classes = 3

    model = SimpleTextClassifier(vocab_size, embedding_dim, hidden_dim, num_classes)

    losses = train_model(model, train_data, epochs=100, lr=0.1)

    # === 4. 测试 ===
    print("\n=== 4. 测试 ===")

    test_texts = ["电影很精彩", "太差", "还行"]
    test_labels = [0, 1, 2]

    test_data = []
    for text, label in zip(test_texts, test_labels):
        indices = processor.process(text)
        test_data.append((indices, label))

    accuracy = evaluate_model(model, test_data)
    print(f"测试准确率: {accuracy:.2%}")

    # === 5. 预测演示 ===
    print("\n=== 5. 预测演示 ===")

    new_texts = ["服务很好", "很差"]

    labels_names = ["褒义", "贬义", "中性"]

    for text in new_texts:
        indices = processor.process(text)
        probs, _ = model.forward(indices)

        predicted = np.argmax(probs)
        print(f"\n输入: '{text}'")
        print(f"预测: {labels_names[predicted]}")
        print(f"概率: 褒={probs[0]:.2f}, 贬={probs[1]:.2f}, 中={probs[2]:.2f}")

    print("\n" + "=" * 50)
    print("情感分析流程总结:")
    print("1. 数据准备：收集标注文本")
    print("2. 预处理：分词、构建词表、向量化")
    print("3. 模型：词嵌入 +序列模型 + 分类层")
    print("4. 训练：优化损失函数")
    print("5. 预测：输入文本 → 情感类别")
    print("=" * 50)
