"""
第11章 计算机视觉进阶 - 图像增广实现
包含：翻转、裁剪、颜色变换、噪声添加
"""

import numpy as np


def horizontal_flip(img):
    """水平翻转

    Args:
        img: 图像数组，形状 (h, w, c)

    Returns:
        翻转后的图像
    """
    return img[:, ::-1, :]


def vertical_flip(img):
    """垂直翻转

    Args:
        img: 图像数组，形状 (h, w, c)

    Returns:
        翻转后的图像
    """
    return img[::-1, :, :]


def random_crop(img, crop_size):
    """随机裁剪

    Args:
        img: 图像数组，形状 (h, w, c)
        crop_size: 裁剪大小 (crop_h, crop_w)

    Returns:
        裁剪后的图像
    """
    h, w = img.shape[:2]
    crop_h, crop_w = crop_size

    if h < crop_h or w < crop_w:
        raise ValueError("裁剪大小不能超过图像大小")

    # 随机选择裁剪起点
    top = np.random.randint(0, h - crop_h + 1)
    left = np.random.randint(0, w - crop_w + 1)

    return img[top : top + crop_h, left : left + crop_w, :]


def center_crop(img, crop_size):
    """中心裁剪

    Args:
        img: 图像数组
        crop_size: 裁剪大小 (crop_h, crop_w)

    Returns:
        裁剪后的图像
    """
    h, w = img.shape[:2]
    crop_h, crop_w = crop_size

    top = (h - crop_h) // 2
    left = (w - crop_w) // 2

    return img[top : top + crop_h, left : left + crop_w, :]


def adjust_brightness(img, factor):
    """调整亮度

    Args:
        img: 图像数组，值范围 [0, 255]
        factor: 亮度因子 (>1 变亮，<1 变暗)

    Returns:
        调整后的图像
    """
    return np.clip(img * factor, 0, 255).astype(img.dtype)


def adjust_contrast(img, factor):
    """调整对比度

    Args:
        img: 图像数组，值范围 [0, 255]
        factor: 对比度因子 (>1 增强，<1 减弱)

    Returns:
        调整后的图像
    """
    mean = img.mean()
    return np.clip((img - mean) * factor + mean, 0, 255).astype(img.dtype)


def add_gaussian_noise(img, mean=0, std=10):
    """添加高斯噪声

    Args:
        img: 图像数组，值范围 [0, 255]
        mean: 噪声均值
        std: 噪声标准差

    Returns:
        添加噪声后的图像
    """
    noise = np.random.normal(mean, std, img.shape)
    return np.clip(img + noise, 0, 255).astype(img.dtype)


def random_rotation(img, max_angle=30):
    """随机旋转（简化实现，使用翻转近似）

    完整实现需要 scipy.ndimage.rotate 或类似库

    Args:
        img: 图像数组
        max_angle: 最大旋转角度

    Returns:
        旋转后的图像（简化版）
    """
    # 简化实现：随机翻转组合
    angle = np.random.randint(-max_angle, max_angle + 1)

    # 正角度：水平翻转
    # 负角度：垂直翻转
    # 实际应用应使用专业的图像旋转库
    if angle > 15:
        return horizontal_flip(img)
    elif angle < -15:
        return vertical_flip(img)
    else:
        return img


def color_jitter(img, brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2)):
    """颜色抖动（随机调整亮度和对比度）

    Args:
        img: 图像数组
        brightness_range: 亮度调整范围
        contrast_range: 对比度调整范围

    Returns:
        调整后的图像
    """
    # 随机亮度
    brightness_factor = np.random.uniform(*brightness_range)
    img = adjust_brightness(img, brightness_factor)

    # 随机对比度
    contrast_factor = np.random.uniform(*contrast_range)
    img = adjust_contrast(img, contrast_factor)

    return img


def compose_augmentations(img, augmentations):
    """组合多种增广方法

    Args:
        img: 图像数组
        augmentations: 增广方法列表，每个元素是 (函数, 参数) 元组

    Returns:
        增广后的图像
    """
    for aug_func, aug_params in augmentations:
        if aug_params is None:
            img = aug_func(img)
        else:
            img = aug_func(img, aug_params)
    return img


if __name__ == "__main__":
    print("=" * 50)
    print("图像增广演示")
    print("=" * 50)

    np.random.seed(42)

    # 创建模拟图像 (16x16 RGB)
    img = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)

    print("\n原始图像:")
    print(f"  形状: {img.shape}")
    print(f"  平均像素值: {img.mean():.1f}")

    # === 1. 翻转 ===
    print("\n=== 1. 翻转 ===")

    h_flipped = horizontal_flip(img)
    v_flipped = vertical_flip(img)

    print(f"水平翻转: 形状 {h_flipped.shape}")
    print(f"垂直翻转: 形状 {v_flipped.shape}")
    print("翻转不改变图像大小，只改变像素位置")

    # === 2. 裁剪 ===
    print("\n=== 2. 裁剪 ===")

    cropped = random_crop(img, (8, 8))
    center_cropped = center_crop(img, (8, 8))

    print(f"随机裁剪: 形状 {cropped.shape}")
    print(f"中心裁剪: 形状 {center_cropped.shape}")
    print("裁剪减小图像大小，保留局部内容")

    # === 3. 颜色变换 ===
    print("\n=== 3. 颜色变换 ===")

    bright = adjust_brightness(img, 1.5)
    dark = adjust_brightness(img, 0.5)
    high_contrast = adjust_contrast(img, 2.0)

    print(f"亮度增强 (factor=1.5): 平均值 {img.mean():.1f} → {bright.mean():.1f}")
    print(f"亮度降低 (factor=0.5): 平均值 {img.mean():.1f} → {dark.mean():.1f}")
    print(
        f"对比度增强 (factor=2.0): 标准差 {img.std():.1f} → {high_contrast.std():.1f}"
    )

    # === 4. 噪声 ===
    print("\n=== 4. 噪声 ===")

    noisy = add_gaussian_noise(img, std=20)

    print("添加高斯噪声 (std=20):")
    print(f"  原图标准差: {img.std():.1f}")
    print(f"  加噪后标准差: {noisy.std():.1f}")
    print("噪声模拟传感器或传输过程中的干扰")

    # === 5. 组合增广 ===
    print("\n=== 5. 组合增广 ===")

    augmentations = [
        (horizontal_flip, None),
        (random_crop, (12, 12)),
        (color_jitter, None),
    ]

    augmented = compose_augmentations(img, augmentations)

    print("组合增广后:")
    print(f"  形状: {augmented.shape}")
    print(f"  平均值: {augmented.mean():.1f}")
    print("\n组合增广可以让模型看到更多样化的数据")

    # === 6. 增广策略 ===
    print("\n=== 6. 增广策略建议 ===")

    print("\n适合自然图像（猫、狗、风景）:")
    print("  - 水平翻转 ✓")
    print("  - 随机裁剪 ✓")
    print("  - 颜色抖动 ✓")
    print("  - 垂直翻转 ✗ (猫不会倒着站)")

    print("\n适合文字图像:")
    print("  - 水平翻转 ✗ (文字镜像后读不了)")
    print("  - 随机裁剪 ✓ (但要保留完整字)")
    print("  - 颜色抖动 ✓")
    print("  - 小角度旋转 ✓")

    print("\n" + "=" * 50)
    print("图像增广总结:")
    print("1. 扩充训练数据，防止过拟合")
    print("2. 让模型应对各种变化")
    print("3. 根据任务选择合适的增广方法")
    print("4. 测试集不增广")
    print("=" * 50)
