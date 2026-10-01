# 🏆 Reranking

> 路径：AI Agent / 📚 RAG
>
> https://app.notion.com/p/3d1776baec498103b334ff1798331166?pvs=204

Retriever 已经把海量知识缩小成一个候选集合，但候选中仍可能存在语义相似却并不能真正回答问题的 Chunk。
这时才值得使用计算成本更高的 Reranker。
## 1. 为什么不一开始就用 Reranker？
如果知识库有百万个 Chunk，让一个更精细的模型逐个判断 Query 与 Chunk 的关系，成本会非常高。
因此常见结构是：
**第一阶段：Retriever → 便宜、快速、扩大候选**
**第二阶段：Reranker → 更精细、更昂贵、缩小候选**
## 2. Reranker 在判断什么？
它通常同时看到 Query 与候选 Chunk，更直接地判断“这个 Chunk 对当前 Query 到底有多相关”。
这和 Embedding Similarity 的判断方式不同：Embedding 更像是在一个预先建立好的语义空间中比较距离；Reranker 则可以对 Query-Chunk 的具体交互进行更细致的判断。
## 3. 为什么只 Rerank Top-K？
因为 Retriever 已经完成第一轮缩小。如果候选有 100 个，精排 100 个仍然可接受；如果直接对整个百万级知识库精排，就失去两阶段检索的意义。
所以候选数量本身就是成本与效果的旋钮。
## 4. Reranker 的硬边界
> **Reranker 只能重新排列“已经被召回”的东西。**
如果相关 Chunk 在第一阶段就被漏掉，Reranker 无法凭空把它找回来。
这就是为什么 Retriever 需要关注 Recall，而 Reranker 才承担更强的 Precision 排序能力。