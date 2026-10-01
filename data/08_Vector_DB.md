# 🗄️ Vector DB

> 路径：AI Agent / 📚 RAG
>
> https://app.notion.com/p/3d1776baec49812a9132dd936953a53d?pvs=204

当 Chunk 已经被 Embedding 成向量，系统还需要一个地方保存这些向量，以及它们对应的原文和元数据。
一个典型记录可以理解为：
**Vector + Chunk + Metadata**
其中 Vector 用于语义检索，Chunk 是最终可以交给 LLM 的知识内容，Metadata 则保存来源、时间、权限、版本、类别等结构化信息。
## 1. 为什么不用普通数据库直接解决全部问题？
普通数据库擅长精确条件查询、事务和结构化数据；向量检索面对的是高维空间中的近邻搜索。随着向量规模增大，逐个计算 Query 与所有 Vector 的距离会越来越昂贵。
因此 Vector DB 通常把向量存储、向量检索、Metadata Filtering、更新删除等能力整合起来，并使用 ANN Index 加速近邻搜索。
## 2. Vector DB 和 Index 不是一回事
这是一个非常重要的层次关系：
> **Vector DB 是“承载系统”；HNSW 是“其中用于加速向量搜索的索引结构”。**
可以把它类比成图书馆和索引：图书馆负责保存书、管理书、提供借阅；索引帮助你更快定位目标。
## 3. 更新知识时发生什么？
如果某个 Document 被修改，受影响的 Chunk 需要重新计算 Embedding，并更新对应的 Vector/Chunk/Metadata；删除的知识也要从检索系统中移除。
因此 RAG 的知识更新通常是 **数据更新 + 重新 Embedding**，而不是重新训练 LLM。
只有当 Embedding Model 本身发生变化时，才可能需要大规模重新 Embedding 整个知识库。