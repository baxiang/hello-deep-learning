"""
第12章 自然语言处理 - 词向量操作演示
包含：词相似度计算、词向量操作、词类比
"""

import numpy as np


def cosine_similarity(v1, v2):
    """计算两个向量的余弦相似度

    Args:
        v1: 向量 1
        v2: 向量 2

    Returns:
        相似度值，范围 [-1, 1]
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0

    return np.dot(v1, v2) / (norm1 * norm2)


def find_similar_words(target_word, word_vectors, top_k=5):
    """找出最相似的词

    Args:
        target_word: 目标词
        word_vectors: 词向量字典
        top_k: 返回前k 个最相似的词

    Returns:
        相似词列表，每个元素是 (词, 相似度)
    """
    if target_word not in word_vectors:
        return []

    target_vec = word_vectors[target_word]
    similarities = {}

    for word, vec in word_vectors.items():
        if word != target_word:
            similarities[word] = cosine_similarity(target_vec, vec)

    # 按相似度排序
    sorted_words = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    return sorted_words[:top_k]


def word_analogy(word_a, word_b, word_c, word_vectors):
    """词类比计算

    计算 a - b + c ≈ ?

    例如：国王 -男 + 女 ≈ 女王

    Args:
        word_a: 词 a
        word_b: 词 b
        word_c: 词 c
        word_vectors: 词向量字典

    Returns:
        最匹配的词
    """
    if (
        word_a not in word_vectors
        or word_b not in word_vectors
        or word_c not in word_vectors
    ):
        return None

    vec_a = word_vectors[word_a]
    vec_b = word_vectors[word_b]
    vec_c = word_vectors[word_c]

    # 计算 a - b + c
    target_vec = vec_a - vec_b + vec_c

    # 找最相似的词（排除a, b, c）
    similarities = {}
    for word, vec in word_vectors.items():
        if word not in [word_a, word_b, word_c]:
            similarities[word] = cosine_similarity(target_vec, vec)

    # 返回最相似的
    sorted_words = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    return sorted_words[:5]


def visualize_word_clusters(word_vectors, words_list):
    """词向量聚类可视化（概念演示）

    Args:
        word_vectors: 词向量字典
        words_list: 要可视化的词列表

    Returns:
        各词之间的相似度矩阵
    """
    n = len(words_list)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if words_list[i] in word_vectors and words_list[j] in word_vectors:
                similarity_matrix[i, j] = cosine_similarity(
                    word_vectors[words_list[i]], word_vectors[words_list[j]]
                )

    return similarity_matrix


if __name__ == "__main__":
    print("=" * 50)
    print("词向量操作演示")
    print("=" * 50)

    np.random.seed(42)

    # === 模拟词向量 ===
    # 实际应用中会使用预训练的 Word2Vec 或 GloVe

    # 动物类词（向量相似）
    animals = {
        "猫": np.array([0.2, 0.8, 0.1, 0.3]),
        "狗": np.array([0.3, 0.7, 0.2, 0.4]),
        "老虎": np.array([0.25, 0.75, 0.15, 0.35]),
        "狮子": np.array([0.22, 0.78, 0.12, 0.32]),
    }

    # 交通工具类词（向量相似）
    vehicles = {
        "汽车": np.array([0.9, 0.1, 0.5, 0.2]),
        "自行车": np.array([0.85, 0.15, 0.45, 0.25]),
        "公交车": np.array([0.88, 0.12, 0.52, 0.18]),
    }

    # 合并词向量
    word_vectors = {**animals, **vehicles}

    print(f"\n词向量库大小: {len(word_vectors)}")
    print(f"向量维度: {len(word_vectors['猫'])}")

    # === 1. 相似度计算 ===
    print("\n=== 1.词相似度计算 ===")

    pairs = [("猫", "狗"), ("猫", "汽车"), ("汽车", "自行车")]

    for w1, w2 in pairs:
        sim = cosine_similarity(word_vectors[w1], word_vectors[w2])
        print(f"  {w1} vs {w2}: {sim:.3f}")

    print("\n解读:")
    print("  相似度高 → 词义相近")
    print("  相似度低 → 词义不同")

    # === 2. 找相似词 ===
    print("\n=== 2. 找相似词 ===")

    for word in ["猫", "汽车"]:
        similar = find_similar_words(word, word_vectors, top_k=3)
        print(f"\n'{word}' 最相似的词:")
        for w, sim in similar:
            print(f"  {w}: {sim:.3f}")

    # === 3. 词类比 ===
    print("\n=== 3. 词类比 ===")

    # 模拟：猫 -老虎 + 公交车 = ?
    # （由于是模拟向量，结果可能不准确，实际应用用真实词向量）
    result = word_analogy("猫", "老虎", "公交车", word_vectors)

    if result:
        print("类比: 猫 -老虎 + 公交车 ≈ ?")
        print("结果:")
        for w, sim in result[:3]:
            print(f"  {w}: {sim:.3f}")

    print("\n实际类比例子（需要真实词向量）:")
    print("  国王 - 男 + 女 ≈ 女王")
    print("  中国 - 北京 + 日本 ≈ 东京")
    print("  走 - 跑 + 看 ≈瞧")

    # === 4. 词聚类 ===
    print("\n=== 4. 词聚类（相似度矩阵）===")

    words_to_check = ["猫", "狗", "汽车", "自行车"]
    sim_matrix = visualize_word_clusters(word_vectors, words_to_check)

    print("\n相似度矩阵:")
    print("      ", end="")
    for w in words_to_check:
        print(f"{w:6s}", end=" ")
    print()

    for i, w in enumerate(words_to_check):
        print(f"{w:4s} ", end="")
        for j in range(len(words_to_check)):
            print(f"{sim_matrix[i, j]:.2f}  ", end=" ")
        print()

    print("\n观察:")
    print("  动物类词之间相似度高")
    print("  交通类词之间相似度高")
    print("  跨类别词相似度低")
    print("  这就是词嵌入的效果！")

    print("\n" + "=" * 50)
    print("词向量总结:")
    print("1. 词向量让计算机理解语义")
    print("2. 相似词的向量接近")
    print("3. 可以做词运算和类比")
    print("4. 核心思想：上下文相似的词，语义相似")
    print("=" * 50)
