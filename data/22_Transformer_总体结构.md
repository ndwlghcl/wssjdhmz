# 🤖 Transformer 总体结构

> 路径：AI / 机器学习学习进度 / 📚 知识点复习索引
>
> https://app.notion.com/p/3bf776baec4981b2b60ede5ce8061457?pvs=204

## 重点是什么
- Transformer 的核心是通过 Attention 建模序列中 token 之间的关系。
- 已经接触 Embedding、Q/K/V、Attention、Multi-Head、Causal Mask、Transformer MLP 等组成部分。
## 当前理解
🟢 **已能把 token/embedding、QKV、Attention、多头输出、Output Projection、Residual 串起主要数据流；细节仍需继续验收**
## 还缺什么
- 从 token → embedding → 位置/位置信息 → QKV → attention → output projection → residual / LayerNorm → MLP 的完整数据流。
- 为什么 Transformer 能高效并行训练。
- 为什么自回归生成时仍需要逐 token 生成。
- V-down / V-up 与具体实现方式的细节，以及不同架构中的差异。