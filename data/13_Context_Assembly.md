# 🧺 Context Assembly

> 路径：AI Agent / 📚 RAG
>
> https://app.notion.com/p/3d1776baec49812c83a4f764c99a1ef4?pvs=204

Retrieval 和 Reranking 得到的是候选知识；LLM 真正需要的是一个经过整理、控制长度、尽量不重复的 Context。
## 1. 为什么需要这一层？
即使 Reranker 已经选出了相关 Chunk，多个 Chunk 之间仍可能重复、顺序混乱、包含不必要内容，或者总长度超过模型上下文预算。
因此通常需要：
- 去重
- 排序
- 截断
- 合并相关片段
- 控制 Context Budget
## 2. 为什么“更多 Context”不一定更好？
加入更多知识可能提高覆盖，但也会带来噪声、上下文成本和注意力分散。最终目标不是让 LLM 看到最多，而是让它看到**足够回答当前 Query 的高价值信息**。
## 3. 它和 Contextual Retrieval 的名字很像，但位置完全不同
**Contextual Retrieval：建库阶段，改善 Chunk 本身。**
**Context Assembly：查询阶段，整理最终交给 LLM 的 Context。**
一个是“提前把知识整理得更可检索”，一个是“现在把检索结果整理得更适合生成”。