# 🕸️ HNSW / ANN

> 路径：AI Agent / 📚 RAG
>
> https://app.notion.com/p/3d1776baec49811882edc788530fb3ad?pvs=204

## 1. 为什么需要 ANN？
最直接的向量检索方法是：Query 与数据库中的每一个 Vector 都计算相似度，再排序取 Top-K。知识库很小时这完全可以接受；规模变大后，逐个比较的代价会越来越高。
ANN（Approximate Nearest Neighbor）因此出现：**不再保证每次都找到数学意义上的绝对最近邻，而是用更低的计算成本找到足够好的近邻。**
它牺牲一部分精确性，换取速度和规模上的可用性。
## 2. HNSW 在做什么？
HNSW（Hierarchical Navigable Small World）是一种常见的 ANN 索引结构。它把向量组织成多层图：高层用于快速跨越较大的区域，低层逐渐细化搜索。
搜索时不是把所有向量都看一遍，而是从合适的节点出发，在图中不断向更接近 Query 的方向移动，再逐步进入更细的层级。
这里的“图”不是知识图谱，也不是 Agent 的工作流图。它是为了**近邻搜索效率**而存在的数据结构。
## 3. HNSW 与 Vector DB 的关系
HNSW 是 Index，不是数据库。实际项目通常直接使用 Vector DB 或向量检索库提供的 HNSW/ANN 实现，而不是自己手写完整 HNSW。
Demo 阶段更值得掌握的是：
**为什么需要 ANN → HNSW 为什么能加速 → 它牺牲了什么 → 它处在哪一层。**
而不是自己实现整个索引。