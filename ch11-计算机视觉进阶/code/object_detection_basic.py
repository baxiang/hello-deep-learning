"""
第11章 计算机视觉进阶 - 目标检测基础
包含：边界框、IoU 计算、锚框生成
"""

import numpy as np


def calculate_iou(box1, box2):
    """计算两个边界框的 IoU

    Args:
        box1: 边界框 (xmin, ymin, xmax, ymax)
        box2: 边界框 (xmin, ymin, xmax, ymax)

    Returns:
        iou: IoU 值，范围 [0, 1]
    """
    # 计算交集区域
    inter_xmin = max(box1[0], box2[0])
    inter_ymin = max(box1[1], box2[1])
    inter_xmax = min(box1[2], box2[2])
    inter_ymax = min(box1[3], box2[3])

    # 交集面积（如果框不重叠则为 0）
    inter_width = max(0, inter_xmax - inter_xmin)
    inter_height = max(0, inter_ymax - inter_ymin)
    inter_area = inter_width * inter_height

    # 各框面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 并集面积
    union_area = area1 + area2 - inter_area

    # IoU
    iou = inter_area / union_area if union_area > 0 else 0

    return iou


def corner_to_center(box):
    """角点格式转中心点格式

    Args:
        box: (xmin, ymin, xmax, ymax)

    Returns:
        (x_center, y_center, width, height)
    """
    xmin, ymin, xmax, ymax = box
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    width = xmax - xmin
    height = ymax - ymin
    return (x_center, y_center, width, height)


def center_to_corner(box):
    """中心点格式转角点格式

    Args:
        box: (x_center, y_center, width, height)

    Returns:
        (xmin, ymin, xmax, ymax)
    """
    x_center, y_center, width, height = box
    xmin = x_center - width / 2
    ymin = y_center - height / 2
    xmax = x_center + width / 2
    ymax = y_center + height / 2
    return (xmin, ymin, xmax, ymax)


def generate_anchor_boxes(image_size, scales, ratios):
    """生成锚框

    Args:
        image_size: 图像大小 (h, w)
        scales: 锚框尺度列表（相对于图像的比例）
        ratios: 锚框宽高比列表

    Returns:
        anchors: 锚框列表，每个锚框为 (xmin, ymin, xmax, ymax)
    """
    h, w = image_size
    anchors = []

    for scale in scales:
        for ratio in ratios:
            # 根据尺度和宽高比计算锚框大小
            anchor_w = w * scale * np.sqrt(ratio)
            anchor_h = h * scale / np.sqrt(ratio)

            # 中心点在图像中心
            x_center = w / 2
            y_center = h / 2

            # 转换为角点格式
            xmin = x_center - anchor_w / 2
            ymin = y_center - anchor_h / 2
            xmax = x_center + anchor_w / 2
            ymax = y_center + anchor_h / 2

            anchors.append((xmin, ymin, xmax, ymax))

    return anchors


def match_anchor_to_ground_truth(anchor, ground_truth, threshold=0.5):
    """判断锚框是否匹配真实框

    Args:
        anchor: 锚框
        ground_truth: 真实框
        threshold: IoU 阈值

    Returns:
        is_match: 是否匹配
        iou: IoU 值
    """
    iou = calculate_iou(anchor, ground_truth)
    is_match = iou >= threshold
    return is_match, iou


def nms(boxes, scores, threshold=0.5):
    """非极大值抑制（NMS）

    去除重叠过多的预测框，只保留最好的

    Args:
        boxes: 预测框列表
        scores: 每个框的置信度
        threshold: IoU 阈值

    Returns:
        keep_indices: 保留的框索引
    """
    if len(boxes) == 0:
        return []

    # 按置信度排序
    order = np.argsort(scores)[::-1]

    keep = []

    while len(order) > 0:
        # 保留置信度最高的框
        idx = order[0]
        keep.append(idx)

        # 计算与其他框的 IoU
        ious = []
        for i in order[1:]:
            ious.append(calculate_iou(boxes[idx], boxes[i]))

        # 去除 IoU 过高的框（重叠过多）
        if len(ious) > 0:
            keep_mask = np.array(ious) < threshold
            order = order[1:][keep_mask]
        else:
            order = []

    return keep


if __name__ == "__main__":
    print("=" * 50)
    print("目标检测基础演示")
    print("=" * 50)

    np.random.seed(42)

    # === 1. IoU 计算 ===
    print("\n=== 1. IoU 计算 ===")

    ground_truth = (100, 100, 200, 200)
    prediction1 = (110, 110, 190, 190)  # 较好
    prediction2 = (50, 50, 150, 150)  # 部分重叠
    prediction3 = (300, 300, 400, 400)  # 完全偏离

    iou1 = calculate_iou(ground_truth, prediction1)
    iou2 = calculate_iou(ground_truth, prediction2)
    iou3 = calculate_iou(ground_truth, prediction3)

    print(f"真实框: {ground_truth}")
    print(f"预测框1: {prediction1}, IoU = {iou1:.3f}")
    print(f"预测框2: {prediction2}, IoU = {iou2:.3f}")
    print(f"预测框3: {prediction3}, IoU = {iou3:.3f}")

    print("\nIoU 解读:")
    print("  > 0.5: 检测成功")
    print("  > 0.7: 检测质量好")
    print("  = 1.0: 完美匹配")

    # === 2. 边界框格式转换 ===
    print("\n=== 2. 边界框格式转换 ===")

    corner_box = (100, 100, 200, 200)
    center_box = corner_to_center(corner_box)

    print(f"角点格式: {corner_box}")
    print(f"中心点格式: {center_box}")

    back_to_corner = center_to_corner(center_box)
    print(f"转换回去: {back_to_corner}")

    # === 3. 锚框生成 ===
    print("\n=== 3. 锚框生成 ===")

    image_size = (100, 100)
    scales = [0.5, 0.8]  # 小和大两种尺度
    ratios = [0.5, 1.0, 2.0]  # 竖、正方、横三种宽高比

    anchors = generate_anchor_boxes(image_size, scales, ratios)

    print(f"图像大小: {image_size}")
    print(f"尺度: {scales}")
    print(f"宽高比: {ratios}")
    print(f"生成锚框数量: {len(anchors)}")

    for i, anchor in enumerate(anchors[:3]):
        print(f"  锚框{i}: {anchor}")

    # === 4. 锚框匹配 ===
    print("\n=== 4. 锚框匹配 ===")

    ground_truth = (30, 30, 70, 70)

    print(f"真实框: {ground_truth}")

    for i, anchor in enumerate(anchors[:3]):
        is_match, iou = match_anchor_to_ground_truth(anchor, ground_truth)
        print(f"  锚框{i}: IoU={iou:.3f}, 匹配={is_match}")

    # === 5. NMS演示 ===
    print("\n=== 5. 非极大值抑制 (NMS) ===")

    # 模拟多个预测框
    boxes = [
        (100, 100, 200, 200),  # 高置信度
        (105, 105, 195, 195),  # 高重叠，应被抑制
        (110, 110, 190, 190),  # 高重叠，应被抑制
        (300, 300, 400, 400),  # 不同位置，应保留
    ]
    scores = [0.9, 0.8, 0.7, 0.6]

    print("预测框:")
    for i, (box, score) in enumerate(zip(boxes, scores)):
        print(f"  框{i}: {box}, 置信度={score}")

    keep_indices = nms(boxes, scores, threshold=0.5)

    print(f"\nNMS 后保留的框索引: {keep_indices}")
    print("NMS 去除重叠过多的框，只保留最好的")

    print("\n" + "=" * 50)
    print("目标检测关键点:")
    print("1. 边界框：用矩形表示物体位置")
    print("2. IoU：衡量两个框的重合程度")
    print("3. 锚框：预设的候选框模板")
    print("4. NMS：去除重叠框，保留最佳")
    print("=" * 50)
