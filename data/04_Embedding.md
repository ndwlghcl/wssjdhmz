# 🧭 Embedding

> 路径：AI Agent / 📚 RAG
>
> https://app.notion.com/p/3d1776baec498170ab0ae22a439a59f0?pvs=204

Embedding 解决的问题不是“怎么搜索”，而是更前面的：**机器怎样把文本变成可以进行相似度计算的对象？**
## 1. 为什么需要向量？
文本不能直接像数字一样计算距离。Embedding Model 把一个 Chunk 映射成一个高维向量；Query 也可以映射到同一个兼容的向量空间。
于是“这两个文本是否语义相关”就被转化为“它们在向量空间里有多接近”。
因此要分清：
> **Embedding = 表示；Retrieval = 查找；Vector DB = 存储与基础设施；Index = 加速查找。**
## 2. Query 和 Chunk 为什么必须在兼容空间？
建库时 Chunk 被 Embedding 成向量；用户查询时 Query 也要使用兼容的 Embedding 方式。只有二者处在可比较的表示空间里，向量相似度才有意义。
这也是为什么更换 Embedding Model 往往意味着重新 Embedding 已有 Chunk：旧向量和新向量通常不再属于同一个可直接比较的空间。
## 3. 相似度到底在比较什么？
常见的是 Cosine Similarity 或 Dot Product。
Cosine 更关注两个向量的方向，而较少受到向量长度影响；Dot Product 同时受到方向和长度影响。当向量已经做 L2 normalization 时，两者在数值上可以对应起来。
真正需要掌握的是：**相似度函数不是“理解文本”的模型本身，它只是利用 Embedding Model 已经产生的表示来计算相似程度。**
## 4. 为什么通常直接用成熟 Embedding Model？
Embedding Model 本身已经通过训练学到了某种语义空间。一般 Demo 和多数业务系统没有必要从零训练自己的模型；只有在领域语言、检索目标或数据分布有特殊需求时，才考虑微调或训练。
## 5. Embedding 与 HNSW 的边界
Embedding 先产生向量；HNSW 不产生向量，也不理解文本。它只负责在已有向量上高效寻找近邻。
因此完整链路是：
**Chunk → Embedding Model → Vector → Index/Search → Candidate Chunks**
而不是“Embedding 就是搜索”。