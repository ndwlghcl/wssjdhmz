# 🪄 Contextual Retrieval

> 路径：AI Agent / 📚 RAG
>
> https://app.notion.com/p/3d1776baec49814ba855fe8ff6670e68?pvs=204

Chunk 被切出来以后，可能出现一种特殊问题：**即使 Chunk 本身不算太小，它单独拿出来仍然缺少背景。**
例如原文前面说“某产品在 2025 年更换了认证方案”，而 Chunk 只有“它在 2025 年进行了调整”。离开原文后，“它”是谁就不明确。
## 1. Contextual Retrieval 在哪里发生？
它属于建库阶段。通常是在 Embedding 前，为 Chunk 补充必要的文档背景，使 Chunk 作为独立检索对象时更容易表达完整语义。
因此链路可以理解成：
**Document → Chunking → Contextualize Chunk → Embedding → Vector DB**
## 2. 它和 Parent-Child 有什么不同？
两者都在缓解上下文问题，但机制不同：
> **Parent-Child：检索小 Child，返回大 Parent。**
>
> **Contextual Retrieval：直接改善 Child 自身的表示/内容，让它离开原文后也更容易被正确检索。**
一个改变的是“检索与返回的粒度关系”，另一个改变的是“Chunk 本身的语义完整性”。
## 3. 为什么可以组合？
一个系统完全可以先给 Child 补上下文，再为 Child 建向量；检索命中后再通过 Parent 提供更完整的原文上下文。
它们解决的是不同层面的缺陷，因此不是互斥方案。