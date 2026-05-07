# Scaled Dot-Product Attention 详解：为什么要除以 $\sqrt{d_k}$？

在 Transformer 的 Attention 机制中，有一个看似简单但至关重要的操作：

```python
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
```

这个除以 `sqrt(d_k)` 的步骤，就是 `"Scaled"` Dot-Product Attention 中 **"Scaled"** 的来源。本文从现象、数学和统计三个层面解释其必要性。

---

## 一、问题背景

Attention 的核心计算是：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

其中：
- $Q \in \mathbb{R}^{n \times d_k}$：Query 矩阵
- $K \in \mathbb{R}^{m \times d_k}$：Key 矩阵
- $d_k$：每个 head 的维度（如 64、128）

如果不做除法，直接计算 $QK^T$，会发生什么？

---

## 二、直观现象：不除会怎样？

### 2.1 点积的数值爆炸

假设 $d_k = 64$，$Q$ 和 $K$ 的每个元素值大约在 $[-1, 1]$ 之间。

两个 $d_k$ 维向量的点积：

$$
\text{score} = q \cdot k^T = \sum_{i=1}^{d_k} q_i k_i
$$

这个求和有 $d_k$ 项，每一项最大可到 $\sim 1$，所以点积结果的范围大约是 $[-d_k, d_k]$，即 **$[-64, 64]$**。

当 $d_k$ 更大时（如 128、256），这个范围还会进一步扩大。

### 2.2 Softmax 饱和与梯度消失

对 score 做 softmax：

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

当 $x_i$ 很大时（比如 64）：
- $e^{64}$ 是一个天文数字
- softmax 会出现**马太效应**：最大的那个值逼近 1，其他所有值都逼近 0
- Attention 分布变成**"one-hot"**（模型只关注某一个位置）

**更严重的是梯度问题**：softmax 的导数在输入很大时会**趋近于 0**。如果 attention score 都是几十上百，反向传播时梯度几乎无法回传，导致**梯度消失**，模型无法学习。

---

## 三、统计解释：为什么恰好是 $\sqrt{d_k}$？

这是一个非常漂亮的数学推导。

### 3.1 假设

假设 $Q$ 和 $K$ 的元素是**独立同分布**的随机变量，满足：
- 均值：$\mathbb{E}[q_i] = \mathbb{E}[k_i] = 0$
- 方差：$\text{Var}(q_i) = \text{Var}(k_i) = 1$

### 3.2 计算点积的方差

$$
\text{score} = \sum_{i=1}^{d_k} q_i k_i
$$

由于 $q_i$ 和 $k_i$ 独立且均值为 0：

$$
\text{Var}(q_i k_i) = \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] - (\mathbb{E}[q_i]\mathbb{E}[k_i])^2 = 1 \cdot 1 - 0 = 1
$$

$d_k$ 项相加（且互不相关）：

$$
\text{Var}(\text{score}) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k
$$

**结论：点积的方差恰好是 $d_k$！**

### 3.3 除以 $\sqrt{d_k}$ 的效果

$$
\text{Var}\left(\frac{\text{score}}{\sqrt{d_k}}\right) = \frac{d_k}{d_k} = 1
$$

**除以 $\sqrt{d_k}$ 后，点积的方差被重新归一化为 1**，数值大小基本不随维度变化，softmax 的输入始终保持在一个健康的范围内。

---

## 四、代码示例

### 4.1 标准实现（PyTorch）

```python
import torch
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: (batch_size, num_heads, seq_len, d_k)
    """
    d_k = Q.size(-1)
    
    # 1. 计算点积注意力分数
    scores = torch.matmul(Q, K.transpose(-2, -1))  # (..., seq_len, seq_len)
    
    # 2. 关键：除以 sqrt(d_k) 进行缩放
    scores = scores / math.sqrt(d_k)
    
    # 3. 可选：应用 mask（如因果 mask）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # 4. Softmax 得到注意力权重
    attn_weights = torch.softmax(scores, dim=-1)
    
    # 5. 加权求和得到输出
    output = torch.matmul(attn_weights, V)
    
    return output, attn_weights

# 示例
batch_size, num_heads, seq_len, d_k = 2, 8, 128, 64
Q = torch.randn(batch_size, num_heads, seq_len, d_k)
K = torch.randn(batch_size, num_heads, seq_len, d_k)
V = torch.randn(batch_size, num_heads, seq_len, d_k)

output, attn = scaled_dot_product_attention(Q, K, V)
```

### 4.2 使用 PyTorch 内置函数

```python
# PyTorch 2.0+ 提供的优化实现
output = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
```

该内置实现会自动处理 scaling 等细节。

---

## 五、拓展：为什么不是除 $d_k$ 而是 $\sqrt{d_k}$？

这是一个常见的疑问。

| 操作 | 效果 | 问题 |
|------|------|------|
| **不除** | 方差 = $d_k$ | 随维度增大而爆炸，softmax 饱和 |
| **除 $d_k$** | 方差 = $1/d_k$ | 标准差 = $1/\sqrt{d_k}$，score 整体趋近于 0 |
| **除 $\sqrt{d_k}$** | 方差 = 1 | ✅ 恰到好处，数值稳定 |

如果除以 $d_k$，随着维度增大：
- score 会整体被压得很小
- softmax 会退化为**均匀分布**（所有位置的权重差不多）
- Attention 失去区分能力，退化为对所有位置平均加权

因此，**除以 $\sqrt{d_k}$ 是在"爆炸"和"坍缩"之间的最佳平衡点**。

---

## 六、可视化对比

假设 $d_k = 64$，对比三种情况的 score 分布：

```
不除 (scores):          范围 [-64, 64]      → softmax 极端尖锐
除 d_k (scores/64):     范围 [-1, 1]        → softmax 过于平滑
除 sqrt(d_k) (scores/8): 范围 [-8, 8]        → softmax 区分度适中 ✅
```

---

## 七、总结

| 问题 | 答案 |
|------|------|
| **为什么要除？** | 防止 $QK^T$ 点积结果随维度 $d_k$ 增大而数值爆炸 |
| **为什么是 $\sqrt{d_k}$？** | 点积方差 = $d_k$，除以 $\sqrt{d_k}$ 后方差 = 1，完美归一化 |
| **不除会怎样？** | Softmax 饱和、Attention one-hot、梯度消失、训练崩溃 |
| **除多了会怎样？** | Score 趋近 0、softmax 均匀分布、Attention 无区分度 |

**"Scaled" Dot-Product Attention 中的 Scale，是 Transformer 能够稳定训练和有效学习的关键设计之一。**

---

## 参考

- Vaswani et al. "Attention Is All You Need", NeurIPS 2017
- 原始 Transformer 论文中 Section 3.2.1 明确给出了除以 $\sqrt{d_k}$ 的统计推导
