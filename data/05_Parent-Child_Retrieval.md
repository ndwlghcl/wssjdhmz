# 🧱 Parent-Child Retrieval

> 路径：AI Agent / 📚 RAG
>
> https://app.notion.com/p/3d1776baec498178869fdcdb8b09af45?pvs=204

Parent-Child Retrieval 是从 Chunk Size 的矛盾自然推导出来的策略。
我们希望 Child 足够小，这样 Retriever 可以精准定位；但又希望最终交给 LLM 的内容足够完整。于是：
> **Child 负责“找到哪里”；Parent 负责“把周围真正有用的上下文带回来”。**
## 1. 为什么不只用小 Chunk？
小 Chunk 检索很精准，但它可能缺少定义、条件、前后文。直接把它交给 LLM，可能导致回答不完整。
## 2. 为什么不只用大 Chunk？
大 Chunk 上下文完整，但一个 Chunk 内可能混合多个主题，使语义表示变得不够聚焦，检索粒度也变粗。
## 3. 典型关系
一个 Parent 可以包含多个 Child：
**Parent（较大上下文） → Child A / Child B / Child C（较小检索单元）**
检索阶段对 Child 做相似度搜索；命中 Child 后，通过关联关系找到对应 Parent，再把 Parent 作为后续 Context 的候选。
## 4. Child 和 Parent 是否都需要 Embedding？
取决于实现方式。如果系统只通过 Child 进行检索，那么通常只需要为 Child 建立检索向量；Parent 主要保存用于返回的上下文。也可以根据具体系统设计让 Parent 也参与检索。
所以不要把“Parent 一定有向量”当成概念定义。核心定义是 **检索粒度与返回粒度被解耦**。
## 5. 与 Reranking 的关系
Parent-Child 可以发生在 Retrieval 的候选形成过程中，也可以根据实现参与后续排序。它不是一个固定的独立 RAG 阶段。
最重要的是理解它解决的来源：**Chunk Size 的粒度—上下文矛盾。**