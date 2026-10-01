# 🔄 Query Transformation

> 路径：AI Agent / 📚 RAG
>
> https://app.notion.com/p/3d1776baec4981b48532e02cbca12689?pvs=204

用户 Query 的目标是表达需求；Retriever 的目标是找到知识。这两个目标并不总是天然一致。
例如用户的问题可能很口语化、包含多个子问题、使用代词，或者缺少检索所需要的关键词。因此在 Retrieval 前可以增加 Query Transformation。
## 1. Rewrite
把原 Query 改写成更明确、更适合搜索的形式。
它通常还是“一问变一问”，只是改变表达。
## 2. Multi-Query / Query Expansion
把一个 Query 扩成多个不同表达或不同角度，再分别检索并合并候选。
它的核心价值是 **Recall**：单个 Query 的表达可能恰好错过某些 Chunk，多几个搜索角度可以扩大候选覆盖。
因此 Multi-Query 本质上是在用更多检索成本换更高召回机会。
## 3. HyDE
HyDE（Hypothetical Document Embeddings）的思路不是直接把原 Query 当作唯一表示，而是先生成一个假想的答案/文档，再利用这个假想文本的语义表示去寻找真实文档。
它适合处理“用户 Query 和知识库文本的表达方式差异较大”的情况。
## 4. 为什么它是可选的？
Query Transformation 会增加模型调用或检索次数，也可能因为错误改写而损伤结果。因此简单、明确的问题没有必要强行经过这一层。
它与 Chunking 的边界也要明确：
> **Chunking 改变知识如何被组织；Query Transformation 改变问题如何进入检索。**