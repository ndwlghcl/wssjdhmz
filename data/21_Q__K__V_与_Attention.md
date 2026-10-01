# 🔑 Q / K / V 与 Attention

> 路径：AI / 机器学习学习进度 / 📚 知识点复习索引
>
> https://app.notion.com/p/3bf776baec498127bcadf2ccd4fdecd3?pvs=204

## 重点是什么
- Q、K、V 是从输入表示经过不同线性变换得到的表示。
- Attention 的核心是根据 Query 与 Key 的匹配程度计算权重，再对 Value 做加权组合。
## 当前理解
🟡 **以前接触并讨论过，需重新验收**
## 还缺什么
- Q/K/V 为什么要分成三个角色。
- attention score、缩放、softmax 的完整计算。
- 一个 token 如何通过 attention 获取其他 token 的信息。
## 2026-08-18 新理解：Value、V-down / V-up 与多头输出
- 每个 attention head 的 Value 会先通过自己的投影进入较低维的 head 空间；可把这部分直觉理解为 **V-down：模型维度 → head 维度**。
- 每个 head 的结果再通过自己的 **V-up** 投回模型空间。
- 多个 head 的 V-up 可以按块拼成一个大的输出投影矩阵；因此多个 head 的低维输出先拼接，再通过整体 output projection 回到 `d_model`。
- 典型维度关系：`n_heads × d_head = d_model`。例如 96 个 head、每个 128 维时：`96 × 128 = 12288`。
- 多头 Attention 完成后，才与原输入做一次残差连接：`X_out = X + AttentionOutput`；不是每个 head 单独做残差。
- 理解到：矩阵乘法可以利用**结合律**重新组织计算顺序：`(AB)C = A(BC)`，但不能随意交换矩阵顺序；`AB` 一般不等于 `BA`。
## 残差连接
- 核心形式：`y = x + F(x)`，可理解为“保留原来的表示 + 这一层学到的修改量”。
- 残差的意义不在于保证向量相加无信息损失，而在于提供恒等路径，使网络可以选择近似不修改输入，并改善深层网络中的信息/梯度传播。
- 当前已能把 Transformer 中的 Attention 输出、Output Projection 与 Residual Add 串起来。