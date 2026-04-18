"""
第12章 自然语言处理 - 文本预处理工具
包含：分词、词表构建、序列处理
"""

import re

import numpy as np


class Tokenizer:
    """分词器"""

    def __init__(self, method="simple"):
        """初始化

        Args:
            method: 分词方法
                - "simple": 简单空格分词
                - "char": 字符级分词
        """
        self.method = method

    def tokenize(self, text):
        """分词

        Args:
            text: 输入文本

        Returns:
            词列表
        """
        if self.method == "simple":
            # 简单空格分词（假设已预处理）
            return text.split()

        elif self.method == "char":
            # 字符级分词
            return list(text)

        else:
            return text.split()

    def detokenize(self, tokens):
        """反分词

        Args:
            tokens: 词列表

        Returns:
            文本
        """
        if self.method == "simple":
            return " ".join(tokens)

        elif self.method == "char":
            return "".join(tokens)

        else:
            return " ".join(tokens)


class Vocabulary:
    """词表管理"""

    def __init__(self, min_freq=1, max_size=None):
        """初始化

        Args:
            min_freq: 最小词频（低于此频率的词不加入词表）
            max_size: 词表最大大小
        """
        self.min_freq = min_freq
        self.max_size = max_size

        # 特殊标记
        self.special_tokens = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}

        self.word2idx = self.special_tokens.copy()
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.word_counts = {}

    def build(self, texts, tokenizer):
        """从文本构建词表

        Args:
            texts: 文本列表
            tokenizer: 分词器

        Returns:
            词表大小
        """
        # 统计词频
        for text in texts:
            tokens = tokenizer.tokenize(text)
            for token in tokens:
                self.word_counts[token] = self.word_counts.get(token, 0) + 1

        # 过滤并构建词表
        filtered_words = [
            (word, count)
            for word, count in self.word_counts.items()
            if count >= self.min_freq
        ]

        # 按频率排序
        filtered_words.sort(key=lambda x: x[1], reverse=True)

        # 限制大小
        if self.max_size:
            filtered_words = filtered_words[: self.max_size - len(self.special_tokens)]

        # 加入词表
        idx = len(self.special_tokens)
        for word, count in filtered_words:
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            idx += 1

        return len(self.word2idx)

    def encode(self, tokens):
        """词转索引

        Args:
            tokens: 词列表

        Returns:
            索引列表
        """
        indices = []
        for token in tokens:
            if token in self.word2idx:
                indices.append(self.word2idx[token])
            else:
                indices.append(self.word2idx["[UNK]"])
        return indices

    def decode(self, indices):
        """索引转词

        Args:
            indices: 索引列表

        Returns:
            词列表
        """
        tokens = []
        for idx in indices:
            if idx in self.idx2word:
                tokens.append(self.idx2word[idx])
            else:
                tokens.append("[UNK]")
        return tokens

    def __len__(self):
        return len(self.word2idx)


class SequenceProcessor:
    """序列处理器"""

    def __init__(self, max_len, pad_idx=0):
        """初始化

        Args:
            max_len: 最大序列长度
            pad_idx: 填充标记索引
        """
        self.max_len = max_len
        self.pad_idx = pad_idx

    def pad(self, sequence, position="post"):
        """填充序列

        Args:
            sequence: 索引序列
            position: 填充位置（"post" 在后面，"pre" 在前面）

        Returns:
            填充后的序列
        """
        if len(sequence) >= self.max_len:
            return sequence[: self.max_len]

        pad_length = self.max_len - len(sequence)

        if position == "post":
            return sequence + [self.pad_idx] * pad_length
        else:
            return [self.pad_idx] * pad_length + sequence

    def truncate(self, sequence, position="post"):
        """截断序列

        Args:
            sequence: 索引序列
            position: 截断位置（"post" 截后面，"pre" 截前面）

        Returns:
            截断后的序列
        """
        if len(sequence) <= self.max_len:
            return sequence

        if position == "post":
            return sequence[: self.max_len]
        else:
            return sequence[-self.max_len :]

    def process(self, sequence, pad_position="post", truncate_position="post"):
        """完整处理：截断 + 填充

        Args:
            sequence: 索引序列
            pad_position: 填充位置
            truncate_position: 截断位置

        Returns:
            处理后的序列
        """
        sequence = self.truncate(sequence, truncate_position)
        sequence = self.pad(sequence, pad_position)
        return sequence


def clean_text(text):
    """文本清洗

    Args:
        text: 输入文本

    Returns:
        清洗后的文本
    """
    # 去除多余空格
    text = re.sub(r"\s+", " ", text)

    # 去除特殊字符（保留中文、英文、数字）
    text = re.sub(r"[^\u4e00-\u9fff\w\s]", "", text)

    # 去除首尾空格
    text = text.strip()

    return text


def create_batches(data, batch_size):
    """创建批次

    Args:
        data: 数据列表，每个元素是 (序列, 标签)
        batch_size: 批次大小

    Returns:
        批次列表
    """
    batches = []

    for i in range(0, len(data), batch_size):
        batch_data = data[i : i + batch_size]

        sequences = [item[0] for item in batch_data]
        labels = [item[1] for item in batch_data]

        batches.append({"sequences": np.array(sequences), "labels": np.array(labels)})

    return batches


if __name__ == "main__":
    print("=" * 50)
    print("文本预处理演示")
    print("=" * 50)

    # === 1. 分词 ===
    print("\n=== 1. 分词 ===")

    tokenizer = Tokenizer(method="simple")

    text = "我喜欢 深度 学习"
    tokens = tokenizer.tokenize(text)

    print(f"原文: '{text}'")
    print(f"分词: {tokens}")
    print(f"反分词: '{tokenizer.detokenize(tokens)}'")

    # 字符级分词
    char_tokenizer = Tokenizer(method="char")
    char_tokens = char_tokenizer.tokenize("我喜欢学习")

    print("\n字符级分词:")
    print("原文: '我喜欢学习'")
    print(f"分词: {char_tokens}")

    # === 2. 词表构建 ===
    print("\n=== 2. 词表构建 ===")

    texts = ["我 喜欢 深度 学习", "这 是 一个 例子", "深度 学习 很 有趣"]

    vocab = Vocabulary(min_freq=1)
    vocab_size = vocab.build(texts, tokenizer)

    print(f"词表大小: {vocab_size}")
    print(f"词表内容: {vocab.word2idx}")

    # 编码解码
    test_tokens = ["我", "喜欢", "编程"]  # "编程"不在词表中
    encoded = vocab.encode(test_tokens)
    decoded = vocab.decode(encoded)

    print("\n编码测试:")
    print(f"原词: {test_tokens}")
    print(f"索引: {encoded}")
    print(f"解码: {decoded}")
    print("注意: '编程'不在词表中，被替换为[UNK]")

    # === 3. 序列处理 ===
    print("\n=== 3. 序列处理 ===")

    processor = SequenceProcessor(max_len=10)

    short_seq = [1, 2, 3]
    long_seq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    padded_short = processor.pad(short_seq)
    truncated_long = processor.truncate(long_seq)

    print(f"短序列: {short_seq}")
    print(f"填充后: {padded_short}")

    print(f"\n长序列: {long_seq}")
    print(f"截断后: {truncated_long}")

    # === 4. 文本清洗 ===
    print("\n=== 4. 文本清洗 ===")

    dirty_text = "这!! 是一个?? 很脏的文本..."

    clean = clean_text(dirty_text)

    print(f"原文: '{dirty_text}'")
    print(f"清洗: '{clean}'")

    # === 5. 批处理 ===
    print("\n=== 5. 批处理 ===")

    data = [
        ([1, 2, 3], 0),
        ([4, 5, 6, 7], 1),
        ([8, 9], 2),
    ]

    # 先处理到相同长度
    processed_data = [(processor.process(item[0]), item[1]) for item in data]

    batches = create_batches(processed_data, batch_size=2)

    print("原始数据:")
    for seq, label in data:
        print(f"  序列: {seq}, 标签: {label}")

    print("\n处理后:")
    for seq, label in processed_data:
        print(f"  序列: {seq}, 标签: {label}")

    print("\n批次:")
    for i, batch in enumerate(batches):
        print(f"  批次{i}:")
        print(f"    序列: {batch['sequences']}")
        print(f"    标签: {batch['labels']}")

    print("\n" + "=" * 50)
    print("文本预处理总结:")
    print("1. 分词：把文本拆成词/字符")
    print("2. 词表：词到索引的映射")
    print("3. 序列处理：填充/截断到固定长度")
    print("4. 文本清洗：去除噪声")
    print("5. 批处理：方便模型训练")
    print("=" * 50)
