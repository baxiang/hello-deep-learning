# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个深度学习入门教程的学习笔记项目（中文），基于《从零开始深度学习》一书。面向零基础开发者，用通俗语言解释深度学习核心概念，配合从零手写的 NumPy 代码示例。

## 参考教材

本项目主要参考两本教材：

| 教材 | 特点 | 适用场景 |
|------|------|----------|
| **《从零开始深度学习》** | 纯 NumPy 实现，概念解释通俗易懂 | 本项目笔记来源，适合零基础入门 |
| **《动手学深度学习》** | PyTorch/TensorFlow 实现，内容更全面 | 进阶学习，涵盖 CNN/RNN/Transformer/NLP 等 |

《动手学深度学习》在线版：https://zh.d2l.ai/

章节对照：
- 本项目 ch01-02 ≈ d2l 第2章（预备知识）
- 本项目 ch03-04 ≈ d2l 第3-4章（线性网络、多层感知机）
- 本项目 ch05 ≈ d2l 第4.7节（反向传播）
- 本项目 ch06 ≈ d2l 第7.5节（批量规范化）、第11章（优化算法）
- 本项目 ch07 ≈ d2l 第6章（卷积神经网络）
- 本项目 ch08 ≈ d2l 第7章（现代 CNN）
- d2l 第8-10章（RNN、注意力、Transformer）← 本项目暂未覆盖
- d2l 第13-15章（计算机视觉、NLP）← 本项目暂未覆盖

## 常用命令

### 环境管理

```bash
# 同步依赖（首次克隆或依赖变更后）
uv sync --all-groups

# 运行代码
uv run python ch01-python基础/01-什么是python/code/hungry.py

# 启动 Jupyter
uv run jupyter notebook
```

### 代码检查

```bash
# Ruff 代码检查
uv run ruff check .

# Ruff 格式化
uv run ruff format .

# 同时执行检查和格式化
uv run ruff check . && uv run ruff format .
```

## 目录结构

每章包含 Markdown 笔记（理论讲解）和 Python 代码示例：

```
ch01-python基础/       # Python、NumPy、Matplotlib 入门
ch02-感知机/           # 感知机原理与逻辑电路实现（AND/NAND/OR/XOR）
ch03-神经网络/         # 神经网络基础、激活函数、前向传播
ch04-神经网络的学习/    # 损失函数、数值微分、梯度下降
ch05-误差反向传播法/    # 计算图、链式法则、反向传播实现
ch06-学习技巧/         # SGD/Adam 优化器、权重初始化、BatchNorm、正则化
ch07-卷积神经网络/      # 卷积层、池化层、CNN 实现
ch08-深度学习进阶/      # 更深网络、超参数调优、全书总结
```

## 代码风格

- 纯 NumPy 实现，不使用深度学习框架（PyTorch/TensorFlow）
- 教学导向：代码简洁清晰，注重可读性而非性能
- 每个代码文件可独立运行，包含测试/演示代码块（在 `if __name__ == "__main__"` 中）
- 函数和类使用中文注释说明用途

## 笔记格式

每篇 Markdown 笔记包含六个部分：
1. 学习目标
2. 核心概念（大白话解释）
3. 原理解析（深入理解）
4. 代码实战（逐行解析）
5. 避坑指南（新手常见错误）
6. 课后思考（动手练习）

## 语言规范

文档和代码注释均为中文。回复用户时使用中文。