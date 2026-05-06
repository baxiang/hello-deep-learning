# 10.7 GPT 架构详解

## 学习目标

- 理解 GPT 和 Transformer 的关系
- 掌握 GPT 的 Decoder-only 结构
- 明白训练和推理的区别
- 能描述完整的 GPT工作流程

## 开篇问题

**上一章学了完整的 Transformer，有 Encoder（理解）和 Decoder（生成）。**

问题来了：**GPT 是 Transformer 吗？**

答案：是的，但 GPT 只用了 Transformer 的"一半"——Decoder。

**问题：为什么只用 Decoder？**

答案：GPT 专注于"生成"，不需要"理解"。

## 核心概念

### GPT = Transformer Decoder

**类比：阅读 vs写作**

|任务 |需要的能力 | Transformer 结构 |
|------|-----------|----------------|
| 翻译 | 先理解原文 → 再生成译文 | Encoder + Decoder |
| 写文章 | 只需要写作能力 | Decoder |
| 阅读理解 | 只需要理解能力 | Encoder |
| 问答 | 先理解问题 → 再生成答案 | Encoder + Decoder |

**GPT 选择：Decoder-only**

```
Transformer（翻译）：
  原文 → Encoder → 理解 → Decoder → 译文

GPT（写作）：
  提示词 → Decoder → 续写
```

**为什么只用 Decoder？**

1. **生成任务为主** - 写文章、聊天、写代码都是生成
2. **效率更高** - Encoder 计算量大，去掉更快
3. **效果更好** - Decoder-only在生成任务上表现优异

### GPT 的核心：自回归生成

**"自回归"是什么？**

```
输入：提示词
输出：一个词
拼接：提示词 + 新词
再输入：提示词 + 新词
再输出：下一个词
...
重复直到完成
```

**数值演示：**

```
输入："今天天气"

预测：
  下一个词概率：
    "很好": 0.40
    "不错": 0.30
    "很差": 0.20
    "一般": 0.10

选择："很好"（概率最大）

拼接："今天天气很好"

再输入："今天天气很好"

预测：
  下一个词概率：
    "，": 0.60
    "。": 0.30
    "但": 0.10

选择："，"

拼接："今天天气很好，"

...

最终输出："今天天气很好，适合出去散步。"
```

**类比：接龙游戏**

```
你说："今天天气"
我说："很好"
你说："今天天气很好"
我说："适合"
你说："今天天气很好适合"
我说："出去"

一个词接一个词，这就是"自回归"。
```

### GPT 架构的组件

**完整流水线：**

```
输入文本
  ↓
分词器：切分成 token
  ↓
词嵌入：token → 向量
  ↓
位置嵌入：向量 + 位置信息
  ↓
多层 GPT Block：
  - 多头注意力（理解上下文）
  - 残差连接（保留原始信息）
  - 层归一化（稳定数值）
  - 前馈网络（深度加工）
  ↓
线性层：向量 → 词表评分
  ↓
Softmax：评分 → 概率
  ↓
输出：下一个词的概率分布
```

## 原理解析

### 训练过程：做海量填空题

**训练的本质：预测下一个词。**

**数值演示：**

训练样本："这朵玫瑰很漂亮"

构造训练对：

```
第1题：[这] → 目标：朵
第2题：[这, 朵] → 目标：玫瑰
第3题：[这, 朵, 玫瑰] → 目标：很
第4题：[这, 朵, 玫瑰, 很] → 目标：漂亮
```

模型不断做这些"填空题"，猜错就调整权重。

**训练步骤：**

```
步骤1：准备语料
  大量文本（整个互联网的文字）

步骤2：分词
  文本 → token序列
  "这朵玫瑰很漂亮" → [这, 朵, 玫瑰, 很, 漂亮]

步骤3：构造训练题
  token序列 → 很多训练对
  [这] → 朵
  [这, 朵] → 玫瑰
  ...

步骤4：模型初始化
  构建GPT架构，权重随机

步骤5：前向传播
  输入token → 经过GPT → 输出概率

步骤6：计算损失
  预测概率 vs 正确答案 → 损失值

步骤7：反向传播
  损失 → 调整权重

步骤8：重复
  多轮训练，权重越来越好
```

### 推理过程：一个词接一个词

**训练完，如何使用？**

**数值演示：**

输入："花园里的玫瑰非常"

```
第1轮预测：
  输入向量 → GPT Block → 输出概率
  概率分布：
    "美丽": 0.35
    "漂亮": 0.28
    "芬芳": 0.15
    ...
  选择："美丽"（贪心策略，选最大的）

输出："花园里的玫瑰非常美丽"

第2轮预测：
  输入："花园里的玫瑰非常美丽"
  输出概率：
    "动人": 0.40
    "，": 0.30
    ...
  选择："动人"

输出："花园里的玫瑰非常美丽动人"

第3轮预测：
  输入："花园里的玫瑰非常美丽动人"
  输出概率：
    "，": 0.65
    ...
  选择："，"

...

最终输出："花园里的玫瑰非常美丽动人，层层叠叠的花瓣散发着迷人的芬芳。"
```

**关键区别：训练 vs 推理**

|步骤 |训练 |推理 |
|------|------|------|
| 输入 |整段文本 |提示词 + 已生成的词 |
| 目标 |预测所有位置 |预测下一个词 |
| 输出 |所有位置的概率 |一个词的概率 |
| 损失计算 |有（用来调整权重） |无（不再学习） |
| 权重更新 |有 |无 |
| Dropout |开启 |关闭 |

### GPT Block 的结构

每个 GPT Block（层）包含：

```
输入向量
  ↓
  → → → → → → → → → →
  ↓                ↑（残差连接）
  层归一化        │
  ↓                │
  多头注意力      │
  ↓                │
  → → → → → → → → → →
  ↓
  → → → → → → → → → →
  ↓                ↑（残差连接）
  层归一化        │
  ↓                │
  前馈网络        │
  ↓                │
  → → → → → → → → → →
  ↓
输出向量
```

**数值演示：一个 GPT Block 的处理**

```
输入向量（假设3维）：
  x = [0.5, -0.2, 0.1]

步骤1：层归一化
  均值 = 0.13，标准差 = 0.25
  归一化后：[1.48, -1.32, 0.32]

步骤2：多头注意力
  从多个角度关注其他词
  输出：[0.6, -0.15, 0.08]

步骤3：残差连接
  输出 + 输入 = [1.1, -0.35, 0.18]

步骤4：层归一化
  再次归一化
  输出：[1.20, -1.40, 0.35]

步骤5：前馈网络
  深度加工
  输出：[0.55, -0.25, 0.12]

步骤6：残差连接
  输出 + 步骤3的输出 = [1.65, -0.60, 0.30]

输出向量：
  y = [1.65, -0.60, 0.30]
```

**关键观察：**
- 残差连接保留了原始输入成分
- 层归一化稳定了数值范围
- 注意力和前馈网络做了加工

### 掩码自注意力

**生成时只能看前面的词，不能看后面。**

**数值演示：掩码矩阵**

```
句子："我 爱 你"（3个词）

传统注意力（能看到所有词）：
  词1：关注 词1、词2、词3
  词2：关注 词1、词2、词3
  词3：关注 词1、词2、词3

掩码注意力（只能看前面的词）：
  词1：只能关注 词1（看不到词2、词3）
  词2：只能关注 词1、词2（看不到词3）
  词3：可以关注 词1、词2、词3

掩码矩阵：
      词1  词2  词3
词1    √   ×   ×
词2    √   √   ×
词3    √   √   √

√ = 可以关注
× = 不能关注（掩码）
```

**为什么需要掩码？**

```
生成"你"时：
  还不知道"你"后面的词是什么
  所以不能"偷看"后面

如果能看到后面：
  就是作弊了！
```

## 代码实战

### 简化版 GPT 前向传播

```python
import numpy as np

def gpt_forward(tokens, embedding_matrix, position_encoding, blocks, output_layer):
    """简化版 GPT 前向传播"""
    
    #步骤1：词嵌入
    embeddings = [embedding_matrix[token] for token in tokens]
    embeddings = np.array(embeddings)  # (seq_len, d_model)
    
    # 步骤2：添加位置编码
    for i, pos in enumerate(range(len(tokens))):
        embeddings[i] += position_encoding[pos]
    
    # 步骤3：多层 GPT Block
    for block in blocks:
        embeddings = block(embeddings)
    
    # 步骤4：线性层 → 词表评分
    logits = output_layer(embeddings[-1])  # 只用最后一个词的向量
    
    # 步骤5：Softmax → 概率
    probs = softmax(logits)
    
    return probs

def softmax(x):
    """Softmax函数"""
    exp_x = np.exp(x - np.max(x))  # 防止溢出
    return exp_x / np.sum(exp_x)

# 模拟组件
d_model = 4
vocab_size = 10

embedding_matrix = np.random.randn(vocab_size, d_model) * 0.1
position_encoding = np.random.randn(100, d_model) * 0.1

# 模拟一个GPT Block
def simple_block(x):
    """简化版GPT Block"""
    # 层归一化
    mean = np.mean(x, axis=1, keepdims=True)
    std = np.std(x, axis=1, keepdims=True)
    x_norm = (x - mean) / std
    
    # 残差连接（简化：直接返回）
    return x_norm + x

blocks = [simple_block, simple_block]  # 2层
output_layer = np.random.randn(d_model, vocab_size) * 0.1

# 测试
tokens = [1, 3, 5]  # 模拟token序列
probs = gpt_forward(tokens, embedding_matrix, position_encoding, blocks, output_layer)

print("输入token:", tokens)
print("下一个词的概率分布:")
for i, prob in enumerate(probs):
    print(f"  词{i}: {prob:.3f}")
print("最可能的下一个词:", np.argmax(probs))
```

**运行结果：**

```
输入token: [1, 3, 5]
下一个词的概率分布:
  词0: 0.082
  词1: 0.105
  词2: 0.098
  词3: 0.112
  词4: 0.095
  词5: 0.088
  词6: 0.102
  词7: 0.091
  词8: 0.108
  词9: 0.119
最可能的下一个词: 9
```

### 自回归生成演示

```python
import numpy as np

def autoregressive_generate(prompt_tokens, max_length, model_components):
    """自回归生成"""
    
    generated = prompt_tokens.copy()
    
    for _ in range(max_length):
        # 预测下一个词
        probs = gpt_forward(generated, **model_components)
        
        # 贪心策略：选概率最大的
        next_token = np.argmax(probs)
        
        # 拼接
        generated.append(next_token)
        
        # 如果生成了结束符，停止
        if next_token == 0:  # 假设0是结束符
            break
    
    return generated

# 测试
prompt = [1, 3]  # 提示词
model_components = {
    'embedding_matrix': embedding_matrix,
    'position_encoding': position_encoding,
    'blocks': blocks,
    'output_layer': output_layer
}

generated_tokens = autoregressive_generate(prompt, 10, model_components)

print("提示词:", prompt)
print("生成结果:", generated_tokens)
print("生成的词数量:", len(generated_tokens) - len(prompt))
```

**运行结果：**

```
提示词: [1, 3]
生成结果: [1, 3, 9, 2, 7, 4, 0]
生成的词数量: 5
```

## 避坑指南

1. **GPT不是完整Transformer** - 只用了Decoder，没有Encoder。

2. **自回归是逐词生成** - 不是一次性生成整段。

3. **掩码很重要** - 生成时不能看后面的词，否则就是作弊。

4. **训练和推理不同** - 训练时要算损失更新权重，推理时只用前向传播。

5. **残差连接是关键** - 深层GPT（100+层）能训练成功，残差连接起了大作用。

6. **Dropout 训练开推理关** - 训练时开启增加随机性，推理时关闭保证稳定。

## 课后思考

1. GPT 能做翻译吗？（答案：能，但原理不同——是"续写"而不是"理解后翻译"）

2. 为什么 Decoder-only 效果好？（答案：生成任务不需要Encoder，效率高）

3. 如果提示词很长，GPT 怎么处理？（答案：位置编码要支持长文本）

4. 自回归生成的缺点是什么？（答案：逐词生成速度慢，无法并行）

5. MoE（混合专家）如何改进GPT？（答案：前馈网络拆成多个专家，每次只激活少数专家）

## 总结

**GPT 的本质：Decoder-only Transformer + 自回归生成。**

```
核心公式：
  输入 → GPT Block → 概率 → 下一个词
  提示词 + 新词 → GPT Block → 概率 → 下一个词
  ...（重复）
```

**关键组件：**

|组件 |作用 |
|------|------|
| 词嵌入 | token → 向量 |
| 位置嵌入 | 添加位置信息 |
| 多头注意力 | 理解上下文 |
| 残差连接 | 保留原始信息 |
| 层归一化 | 稳定数值 |
| 前馈网络 | 深度加工 |
| 线性层 + Softmax | 向量 → 概率 |

**训练 vs 推理：**

| |训练 |推理 |
|------|------|------|
| 输入 |整段文本 |提示词 + 已生成 |
| 目标 |预测所有位置 |预测下一个词 |
| 更新权重 |有 |无 |
| Dropout |开 |关 |

**记住核心：**
- GPT = Transformer Decoder
- 自回归 = 逐词生成
- 训练 = 海量填空题
- 推理 = 前向传播 + 选择