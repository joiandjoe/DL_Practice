---
name: 小型Transformer语言模型实现
overview: 实现一个极小的Transformer语言模型（2-4层，64-128隐藏维度），包含完整的训练和推理功能，使用PyTorch在Jupyter Notebook中实现。
todos:
  - id: data_prep
    content: 实现数据准备：创建简单的文本数据集，实现tokenization和数据加载器
    status: completed
  - id: attention
    content: 实现MultiHeadAttention组件：包含QKV投影、scaled dot-product attention、causal mask
    status: completed
  - id: ffn
    content: 实现FeedForward网络组件
    status: completed
  - id: transformer_block
    content: 实现TransformerBlock：组合attention和FFN，使用pre-norm和残差连接
    status: completed
  - id: full_model
    content: 实现完整的TransformerLM模型：embedding、位置编码、多个transformer blocks、输出层
    status: completed
  - id: training
    content: 实现训练循环：前向传播、损失计算、反向传播、优化器更新
    status: completed
  - id: inference
    content: 实现推理/生成函数：自回归生成、支持temperature和采样策略
    status: completed
  - id: testing
    content: 添加测试和示例：训练模型、生成文本示例、可视化结果
    status: completed
isProject: false
---

# 小型Transformer语言模型实现计划

## 概述

在 `CS336-lec5.ipynb` 中实现一个极小的Transformer语言模型，用于文本生成任务。模型将采用标准的decoder-only架构（类似GPT），包含完整的训练和推理功能。

## 架构设计

### 模型组件

1. **Token Embedding**: 将token ID转换为向量
2. **Positional Encoding**: 使用可学习的位置编码（简单实现）或正弦位置编码
3. **Transformer Block** (2-4层):

   - Pre-norm Layer Normalization
   - Multi-Head Self-Attention
   - Feed-Forward Network (FFN)
   - Residual connections

4. **Output Layer**: 将隐藏状态映射到词汇表大小

### 模型超参数（极小模型）

- `vocab_size`: 根据数据集确定（字符级约100-200，词级约1000-5000）
- `d_model`: 128（隐藏维度）
- `n_layers`: 3（Transformer层数）
- `n_heads`: 4（注意力头数，确保 d_model % n_heads == 0）
- `d_ff`: 512（FFN中间层维度，通常为d_model的4倍）
- `max_seq_len`: 128（最大序列长度）
- `dropout`: 0.1（可选，用于正则化）

## 实现步骤

### 1. 数据准备

- 使用简单的文本数据集（如Tiny Shakespeare、简单中文文本或自动生成的序列数据）
- 实现字符级或词级tokenization
- 创建数据加载器，支持batch处理和序列截断/填充

### 2. 模型实现

在notebook中按顺序实现：

**2.1 基础组件**

- `MultiHeadAttention`: 实现多头自注意力机制
  - Q, K, V投影
  - Scaled dot-product attention
  - Causal mask（用于语言模型）
  - 多头拼接和输出投影

- `FeedForward`: 实现前馈网络
  - 两层线性变换
  - 激活函数（GELU或ReLU）

- `TransformerBlock`: 组合attention和FFN
  - Pre-norm架构
  - Residual connections

**2.2 完整模型**

- `TransformerLM`: 主模型类
  - Token embedding
  - Positional encoding
  - N个Transformer blocks
  - Layer normalization
  - Output projection到词汇表

### 3. 训练流程

- 实现训练循环：
  - 前向传播
  - 计算交叉熵损失（next token prediction）
  - 反向传播
  - 优化器更新（AdamW）
  - 学习率调度（可选warmup）
- 添加训练监控（loss记录、定期打印）

### 4. 推理/生成

- 实现自回归生成函数：
  - 给定prompt，逐步生成token
  - 支持temperature sampling和top-k/top-p采样
  - 支持最大生成长度限制

### 5. 评估和测试

- 在训练集上测试模型收敛
- 使用训练好的模型进行文本生成示例
- 可视化训练loss曲线

## 文件结构

```
CS336-lec5.ipynb
├── 数据准备和预处理
├── 模型组件定义
│   ├── MultiHeadAttention
│   ├── FeedForward  
│   ├── TransformerBlock
│   └── TransformerLM
├── 训练循环
├── 推理/生成函数
└── 测试和示例
```

## 技术细节

### 关键实现点

1. **Causal Mask**: 确保模型只能看到当前位置之前的token
2. **Pre-norm**: 在attention和FFN之前应用LayerNorm（符合lec3的现代实践）
3. **位置编码**: 使用可学习的位置嵌入（简单）或正弦位置编码
4. **损失函数**: 使用交叉熵损失，预测下一个token

### 数据集选择

推荐使用字符级的小型文本数据集，例如：

- 自动生成的简单序列模式（如重复模式、计数等）
- 或使用Tiny Shakespeare等公开小数据集
- 便于快速训练和验证模型功能

## 预期输出

- 完整的可训练Transformer模型
- 训练脚本和训练过程可视化
- 文本生成示例，展示模型学习到的模式