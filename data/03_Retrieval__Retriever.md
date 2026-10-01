# 🔎 Retrieval / Retriever

> 路径：AI Agent / 📚 RAG
>
> https://app.notion.com/p/3d1776baec4981388d38e98021885e6d?pvs=204

RAG 最核心的“找知识”发生在 Retrieval。
但要特别避免一个常见误解：**Retrieval ≠ Vector Search。** Vector Search 只是 Retrieval 的一种实现路线。
## 1. 为什么需要 Retriever？
LLM 不可能每次回答问题都把整个知识库读一遍。Retriever 的任务就是利用 Query，从海量知识中筛出一个较小的候选集合。
因此第一阶段更像一个“粗筛”：它的首要目标通常不是判断谁最完美，而是尽量别漏掉真正有用的知识。
## 2. Vector Search 路线
**Query → Query Embedding → Similarity → ANN/HNSW → Top-K**
Query 和 Chunk 被映射到兼容向量空间后，系统根据相似度寻找近邻。
## 3. Keyword Search 路线
**Query → Terms → Inverted Index → BM25 等评分 → Top-K**
它不需要向量表示。对于错误码、编号、精确名称、专有词等内容，词项本身就是强信号。
## 4. Hybrid Search
Vector Search 擅长语义相似；Keyword Search 擅长精确词项。两者的失败模式具有互补性，因此可以把它们组合起来。
Hybrid Search 是一种 Retrieval Strategy，而不是 RAG 流程中固定的“下一步”。
## 5. Recall 和 Precision
设真正相关的 Chunk 集合为 Relevant，Retriever 返回的候选集合为 Retrieved：
**Recall = 被召回的相关内容 / 所有相关内容**
**Precision = 被召回的相关内容 / 所有被召回内容**
Recall 高说明“漏得少”；Precision 高说明“召回结果里杂质少”。
二者存在典型张力：扩大候选集合通常更容易提高 Recall，但也可能带入更多噪声，从而降低 Precision。
## 6. 为什么第一阶段偏向 Recall？
因为存在一个不可逆的边界：
> **没有被 Retriever 召回的 Chunk，后面的 Reranker 根本没有机会判断。**
这就是两阶段检索的关键逻辑：第一阶段负责把“可能有用的东西”尽量带进候选集；第二阶段再花更多计算量精细判断。
## 7. Top-K 为什么重要？
K 太小：候选集合小、计算便宜，但可能漏掉真正相关内容。
K 太大：Recall 可能提高，但噪声、Reranking 成本和最终 Context Cost 都会上升。
所以 Top-K 不是一个固定数字，而是系统在 Recall、Precision、Latency、Cost 之间做的工程权衡。
## 8. F1
F1 用调和平均综合 Precision 和 Recall：
`F1 = 2PR / (P + R)`
这里值得理解的不是公式本身，而是它表达的思想：**一个候选集合不能只靠“覆盖很多”或“很干净”中的一边来评价。**
## 9. Retrieval 与 Reranking 的关系
可以把整个过程理解成：
**海量知识 → Retriever 粗召回 → 候选集合 → Reranker 精排 → 少量高价值 Context**
Retriever 和 Reranker 不是重复做同一件事，而是在不同候选规模下承担不同精度与成本目标。