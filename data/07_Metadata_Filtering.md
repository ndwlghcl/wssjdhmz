# 🧩 Metadata Filtering

> 路径：AI Agent / 📚 RAG
>
> https://app.notion.com/p/3d1776baec49818888c3c679ca7fba08?pvs=204

向量相似度或关键词相关性回答的是：**“这个内容与 Query 有没有关系？”**
真实系统往往还有另一类问题：**“即使它相关，它是否符合当前检索条件？”**
例如：
- 只允许搜索 2026 年资料
- 只看当前版本
- 只看某个部门
- 只返回用户有权限访问的文档
- 只搜索某种来源或类别
这些条件可以通过 Metadata Filtering 表达。
## 1. Retrieval 与 Filtering 的区别
> **Retrieval：哪些内容可能相关？**
>
> **Metadata Filtering：哪些内容满足结构化约束？**
它们不是同一个判断维度。
## 2. 为什么 Filter 会影响 Recall？
假设真正相关的 Chunk 被正确检索到了，但它的 Metadata 不满足 Filter，那么它仍然会被排除。
因此过于严格的 Filter 或错误 Metadata 都可能让 Recall 下降。
## 3. Pre-filter / Post-filter
Filter 的具体位置并没有一个所有系统都统一的答案：有的系统会先过滤候选空间，再做向量检索；有的实现会先检索再过滤；也有系统把过滤条件直接融入搜索过程。
所以学习重点不是死记“Filter 一定发生在 Retrieval 前/后”，而是理解：**它会改变允许进入结果的候选集合。**
## 4. 与 Query Transformation 的区别
Query Transformation 改变“问题怎样被搜索”；Metadata Filtering 限定“哪些数据可以被考虑”。
前者是检索表达策略，后者是结构化约束。