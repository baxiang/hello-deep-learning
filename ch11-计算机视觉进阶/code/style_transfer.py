"""
第11章 计算机视觉进阶 - 风格迁移（概念演示）
包含：格拉姆矩阵、损失计算

注意：完整实现需要 PyTorch，这里只演示核心概念
"""

import numpy as np


def gram_matrix(features):
    """计算格拉姆矩阵

    格拉姆矩阵捕捉特征通道之间的相关性，
    用于表示图像的"风格"

    Args:
        features: 特征图，形状 (h, w, c) 或 (c, h, w)

    Returns:
        gram: 格拉姆矩阵，形状 (c, c)
    """
    # 假设形状是 (h, w, c)
    h, w, c = features.shape

    # 展平空间维度: (h*w, c)
    features_flat = features.reshape(h * w, c)

    # 计算格拉姆矩阵: F^T @ F / (h*w)
    # 每个元素表示两个特征通道的相关程度
    gram = np.dot(features_flat.T, features_flat) / (h * w)

    return gram


def content_loss(content_features, generated_features):
    """内容损失

    保持生成图与内容图的"内容"相似
    内容 = 物体的形状和布局，用特征表示

    Args:
        content_features: 内容图的特征
        generated_features: 生成图的特征

    Returns:
        loss: 内容损失值
    """
    return np.mean((content_features - generated_features) ** 2)


def style_loss(style_gram, generated_gram):
    """风格损失

    让生成图学习风格图的"风格"
    风格 = 纹理、颜色分布、笔触，用格拉姆矩阵表示

    Args:
        style_gram: 风格图的格拉姆矩阵
        generated_gram: 生成图的格拉姆矩阵

    Returns:
        loss: 风格损失值
    """
    return np.mean((style_gram - generated_gram) ** 2)


def total_variation_loss(img):
    """总变分损失（平滑损失）

    减少生成图的噪声，使图像更平滑

    Args:
        img: 图像，形状 (h, w, c)

    Returns:
        loss: 总变分损失
    """
    # 水平方向的差异
    h_diff = np.mean((img[:, 1:, :] - img[:, :-1, :]) ** 2)

    # 垂直方向的差异
    v_diff = np.mean((img[1:, :, :] - img[:-1, :, :]) ** 2)

    return h_diff + v_diff


def total_loss(
    content_loss_val,
    style_loss_val,
    tv_loss_val,
    content_weight=1.0,
    style_weight=1000.0,
    tv_weight=1.0,
):
    """总损失

    Args:
        content_loss_val: 内容损失
        style_loss_val: 风格损失
        tv_loss_val: 总变分损失
        content_weight: 内容损失权重
        style_weight: 风格损失权重
        tv_weight: 总变分损失权重

    Returns:
        total: 总损失
    """
    total = (
        content_weight * content_loss_val
        + style_weight * style_loss_val
        + tv_weight * tv_loss_val
    )
    return total


class StyleTransferDemo:
    """风格迁移演示（概念版）"""

    def __init__(self):
        """初始化"""
        # 模拟 VGG 特征提取器
        np.random.seed(42)

    def extract_features(self, img, layer_idx=0):
        """模拟特征提取

        实际应用中用预训练的 VGG 网络

        Args:
            img: 输入图像
            layer_idx: 层索引（不同层提取不同级别的特征）

        Returns:
            features: 提取的特征
        """
        # 模拟：简单的随机变换
        h, w, c = img.shape
        feature_c = 16  # 特征通道数

        # 模拟不同层的特征
        np.random.seed(layer_idx * 100 + int(img.mean()))
        features = np.random.randn(h // 2, w // 2, feature_c)

        return features


if __name__ == "__main__":
    print("=" * 50)
    print("风格迁移概念演示")
    print("=" * 50)

    np.random.seed(42)

    # === 1. 格拉姆矩阵 ===
    print("\n=== 1. 格拉姆矩阵 ===")

    # 模拟特征图
    features = np.random.randn(8, 8, 16)  # 8x8 空间，16个特征通道

    print(f"特征图形状: {features.shape}")
    print("特征图 = CNN提取的图像特征")

    gram = gram_matrix(features)

    print(f"\n格拉姆矩阵形状: {gram.shape}")
    print("格拉姆矩阵 = 特征通道之间的相关性")
    print("\n格拉姆矩阵示例（前4x4）:")
    print(gram[:4, :4])

    print("\n解读:")
    print("gram[i,j] = 通道i 和通道 j 的相关性")
    print("正相关 → 这两种特征经常一起出现")
    print("负相关 → 这两种特征很少同时出现")
    print("这种'相关性模式'就代表了图像的风格")

    # === 2. 损失计算 ===
    print("\n=== 2. 损失计算 ===")

    # 模拟三张图的特征
    content_features = np.random.randn(8, 8, 16)
    style_features = np.random.randn(8, 8, 16)
    generated_features = np.random.randn(8, 8, 16)

    # 计算格拉姆矩阵
    style_gram = gram_matrix(style_features)
    generated_gram = gram_matrix(generated_features)

    # 计算损失
    c_loss = content_loss(content_features, generated_features)
    s_loss = style_loss(style_gram, generated_gram)

    print(f"内容损失: {c_loss:.4f}")
    print(f"风格损失: {s_loss:.4f}")

    # === 3. 权重调节 ===
    print("\n=== 3. 权重调节 ===")

    for style_weight in [1, 100, 1000, 10000]:
        total = total_loss(c_loss, s_loss, 0, style_weight=style_weight)
        print(f"风格权重={style_weight:5d}: 总损失={total:.2f}")

    print("\n权重解读:")
    print("风格权重小 → 更像原图（保留内容）")
    print("风格权重大 → 更像风格图（学习风格）")

    # === 4. 多层特征 ===
    print("\n=== 4. 多层特征 ===")

    demo = StyleTransferDemo()

    # 模拟图像
    img = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)

    print("不同层提取的特征:")
    for layer in range(3):
        feat = demo.extract_features(img, layer)
        gram = gram_matrix(feat)
        print(f"  层{layer}: 特征形状{feat.shape}, Gram形状{gram.shape}")

    print("\n多层风格损失:")
    print("底层特征 → 细粒度风格（纹理）")
    print("高层特征 → 粗粒度风格（整体色调）")
    print("多层组合 → 更丰富的风格学习")

    # === 5. 风格迁移流程 ===
    print("\n=== 5. 风格迁移流程 ===")

    print("\n实际流程:")
    print("1. 加载内容图和风格图")
    print("2. 用预训练 VGG 提取多层特征")
    print("3. 计算内容图和风格图的格拉姆矩阵")
    print("4. 初始化生成图（可以是内容图副本）")
    print("5. 循环优化:")
    print("   a. 提取生成图特征")
    print("   b. 计算内容损失和风格损失")
    print("   c. 计算总损失和梯度")
    print("   d. 更新生成图的像素（不是网络参数！）")
    print("6. 输出最终生成图")

    print("\n关键点:")
    print("- 优化的是生成图的像素，不是网络参数")
    print("- VGG 网络参数固定，只用于提取特征")
    print("- 需要预训练的深度网络（如VGG19）")
    print("- 实际实现推荐用 PyTorch")

    print("\n" + "=" * 50)
    print("风格迁移总结:")
    print("内容损失: 保持物体形状和布局")
    print("风格损失: 学习纹理和笔触")
    print("格拉姆矩阵: 捕捉特征相关性")
    print("权重调节: 控制风格化程度")
    print("=" * 50)
