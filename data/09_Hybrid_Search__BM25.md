# 🔀 Hybrid Search / BM25

> 路径：AI Agent / 📚 RAG
>
> https://app.notion.com/p/3d1776baec4981b0a028c8e3b8fb687d?pvs=204

## 1. 为什么纯向量搜索不够？
Vector Search 擅长理解语义相近，但对于错误码、SKU、API 名称、专有词、精确编号等内容，字面一致本身就是非常重要的信号。
如果 Query 与知识库中某个 Chunk 只有一个关键术语完全一致，而另一个 Chunk 在整体语义上“很像”，纯向量排序不一定把精确匹配放在最前面。
## 2. 为什么纯关键词搜索也不够？
关键词搜索依赖词项重合。用户说“怎么撤销订单”，文档可能写“取消购买请求”，两者意思接近，但字面词不同，关键词检索可能漏掉它。
因此两条路线形成互补：
**Vector Search → 解决“说法不同但意思相近”**
**Keyword Search → 解决“必须匹配具体词项”**
## 3. BM25 在做什么？
BM25 是经典的词项相关性评分方法。直觉上，它综合考虑：Query 中的词是否出现在文档中、词在文档中的稀有程度、文档长度等因素。
不需要记住完整公式才能理解它。对 RAG 来说，更重要的是知道它属于 Keyword Retrieval 这条路线。
## 4. Hybrid 怎么组合？
常见思路是分别得到 Vector Search 和 Keyword Search 的候选/分数，再进行融合，形成更稳健的候选集合，然后可以继续 Reranking。
具体融合算法和分数归一化方式会因系统而异，因此 Hybrid Search 应理解为 **Retrieval Strategy**，而不是一个固定的新流程阶段。