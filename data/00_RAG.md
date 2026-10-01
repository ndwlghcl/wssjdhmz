# 📚 RAG

> 路径：AI Agent
>
> https://app.notion.com/p/3d1776baec498123b766d160af1bc7c0?pvs=204

# RAG：让 LLM 在需要时找到外部知识
RAG（Retrieval-Augmented Generation）最核心的思想不是“把文档放进向量数据库”，而是把**知识获取**从 LLM 的参数记忆中拆出来：当用户提出问题时，系统先从外部知识中找到当前真正需要的内容，再把这些内容交给 LLM 生成答案。
所以理解 RAG 最好的方式不是背一串名词，而是沿着一个问题往下追：
> **知识原本是什么样？为什么不能直接拿来搜？→ 怎样把它变成可检索的数据？→ 用户提问后怎么找到相关知识？→ 找到很多候选后怎么减少错误？→ 最后到底给 LLM 看什么？→ 🔄 RAG 知识更新**
---
## 🗺️ 先看全貌：RAG 是“离线准备 + 在线寻找 + 生成”的系统
```plain text
                    ┌──────────── 离线建库 ────────────┐
                    │                                   │
Document → Chunking →（可选 Contextual Retrieval）→ Embedding
                    │                                   │
                    └──────────────────────────────→ Vector DB + Index
                                                        │
                                                        │
用户 Query ──→（可选 Query Transformation）────────────┤
                                                        ↓
                              ┌────────── Retrieval ──────────┐
                              │                               │
                              │ Vector Search     Keyword Search
                              │   ↓                    ↓      │
                              │ Query Embedding      BM25/倒排索引 │
                              │   ↓                    ↓      │
                              │ ANN / HNSW          词项匹配    │
                              │        ↘          ↙            │
                              │         Hybrid（可选）         │
                              └────────────┬──────────────────┘
                                           ↓
                                  Metadata Filtering
                                           ↓
                                      Reranking
                                           ↓
                                  Context Assembly
                                           ↓
                                  Query + Context
                                           ↓
                                          LLM
                                           ↓
                                         Answer
```
这张图里最重要的是：**它不是一条永远固定的直线。** Retrieval 本身存在不同路线；Query Transformation、Parent-Child、Contextual Retrieval、Hybrid Search 等，是针对特定问题插入的策略，而不是凭空多出来的“第 N 步”。
---
# 一、离线建库：先把“原始知识”变成“可检索知识”
用户还没有提问之前，系统已经需要做一轮准备。原因很简单：原始文档是给人阅读的，不是直接为大规模检索设计的。
## 1. Document → Chunking：先决定“检索的基本单位”
原始 Document 往往很长。如果把整篇文档当成一个检索单位，那么用户只问其中一个局部问题时，系统只能把整篇文档当成候选；文档的整体语义会掩盖局部信息，返回给 LLM 时也会带进大量无关内容。
因此需要 ✂️ Chunking：把 Document 划成较小的 Chunk，让“知识库里什么东西可以被单独找出来”变得更细。
但这里马上出现一个矛盾：
- Chunk 小 → 检索粒度细，更容易精准定位；但上下文容易断裂。
- Chunk 大 → 上下文更完整；但无关内容更多，检索粒度变粗，成本也会上升。
所以 **Chunk Size 不是一个应该死记的最佳数字，而是“检索粒度 ↔ 上下文完整性 ↔ 成本”的权衡。**
这个矛盾后来会自然长出 🧱 Parent-Child Retrieval：如果“用于定位”的大小和“用于回答”的大小不必相同，那么就让小 Child 负责找到位置，大 Parent 负责提供更完整的上下文。
## 2. Chunk → Contextual Retrieval（可选）→ Embedding：解决“这个 Chunk 单独拿出来还看得懂吗？”
Chunk 切出来以后还有另一个问题：**它可能失去原文背景。**
例如原文前面已经说明“某产品在 2025 年更换了认证方案”，而当前 Chunk 只有“它在 2025 年进行了调整”。人读完整文档知道“它”是谁，但 Chunk 单独进入检索系统后，信息是不完整的。
这时可以在建库阶段使用 🪄 Contextual Retrieval，先给 Chunk 补充必要背景，再进入 Embedding。
注意它和 Parent-Child 解决的不是同一个问题：
> **Parent-Child 改变的是“检索粒度和返回粒度的关系”；Contextual Retrieval 改善的是“Chunk 自身的语义完整性”。**
两者可以同时存在。
接下来进入 🧭 Embedding。
Embedding 解决的是另一个层次的问题：**文本本身不好直接进行语义上的“距离计算”，所以需要一种机器可比较的表示。**
它把 Chunk 映射到向量空间。以后 Query 也要进入兼容的向量空间，这样系统才能比较 Query 与 Chunk 的语义关系。
因此要严格区分：
> **Embedding 负责“怎么表示”；Retrieval 负责“怎么找”；Vector DB 负责“怎么存和提供基础设施”；Index 负责“怎么让搜索更快”。**
## 3. Vector DB + Index：知识已经变成向量，还得能高效地找
建库时最终保存的通常不只是一个向量，而是一组关联数据：
**Chunk + Vector + Metadata**
Vector DB 负责保存这些数据，并提供查询、过滤、更新等能力。
但如果知识库里有百万甚至更多向量，每次 Query 都和全部向量逐个比较，会越来越慢。因此向量数据库通常会使用 ANN（Approximate Nearest Neighbor）之类的近似近邻搜索方法，其中一种常见索引结构就是 HNSW。
这里特别容易混淆：**HNSW 不是 Vector DB。** Vector DB 是承载数据和检索能力的基础设施；HNSW 是其中可以用于加速向量近邻搜索的 Index。
---
# 二、在线查询：用户真正提问以后，系统才开始“找知识”
离线阶段解决的是“知识怎么准备”；在线阶段解决的是“这个问题现在需要哪些知识”。
## 1. Query → Query Transformation（可选）
用户表达问题是为了交流，不一定是为了检索。一个 Query 可能口语化、含义不清，或者一个问题里实际上包含多个搜索角度。
因此可以先经过 🔄 Query Transformation。
它有不同思路：
- **Rewrite**：把原问题改写成更适合搜索的表达。
- **Multi-Query / Query Expansion**：把一个问题扩成多个搜索角度，用多个 Query 增加 Recall。
- **HyDE**：先生成假想的答案/文档，再利用它的语义表示去寻找真实知识。
它们不是必经步骤。简单 Query 可以直接检索；只有当原始 Query 本身可能限制检索效果时，才值得增加这一层。
---
# 三、Retrieval：真正的“找知识”，但这里其实有不同路线
这一层是整个 RAG 最容易被讲成一条错误直线的地方。
很多入门材料会写成：
**Query → Embedding → Vector DB → Top-K**
但这只是 **Vector Search** 这一条路线。RAG 的 Retrieval 并不等于向量搜索。
## 路线 A：Vector Search
**Query → Query Embedding → 向量相似度 → ANN / HNSW → Top-K**
它擅长处理“措辞不同但意思相近”的问题。Query 与 Chunk 被放入同一个语义空间以后，系统可以根据向量之间的相似程度寻找候选。
HNSW 就位于这条路线的底层：它不是在判断“这个 Chunk 是否回答问题”，而是在帮助系统更快找到向量空间里的近邻。
## 路线 B：Keyword Search
**Query → 关键词/词项 → 倒排索引 → BM25 等评分 → Top-K**
它不需要把 Query 转成向量。对于错误码、编号、精确产品名、专有词等场景，精确词项本身就是非常重要的信号。
## 为什么需要 Hybrid Search？
两条路线的失败方式正好有互补性：
- Vector Search 能理解语义，但可能把一个“意思相近”的结果排在必须精确匹配的词项前面。
- Keyword Search 能精确匹配词，但可能找不到“说法不同、意思相同”的内容。
因此 🔀 Hybrid Search / BM25 是把两条 Retrieval 路线组合起来，而不是“Vector Search 后面的下一步”。
---
# 四、Retrieval 的第一目标：先别漏掉真正有用的知识
无论采用 Vector、Keyword 还是 Hybrid，第一阶段面对的都是海量知识，因此通常要先做**粗召回**。
🔎 Retrieval / Retriever 的核心任务可以理解成：**把整个知识库缩小成一个值得进一步处理的候选集合。**
这里最重要的是 Recall 与 Precision 的张力。
- **Recall**：真正相关的内容，有多少被召回了？
- **Precision**：召回出来的内容，有多少是真的相关？
第一阶段通常更偏向 Recall。因为：
> **如果一个真正相关的 Chunk 根本没有进入候选集合，后面的 Reranker 再强也无法把它找回来。**
这也是 Top-K 存在的原因：K 太小，候选少但容易漏；K 太大，Recall 可能提高，却会带来更多噪声、计算成本和上下文成本。
F1 可以用调和平均把 Precision 与 Recall 合并起来衡量：`F1 = 2PR / (P + R)`。这里真正值得理解的是：**Recall 与 Precision 不是两个孤立指标，而是在描述同一个候选集合的两个侧面。**
---
# 五、Metadata Filtering：相关还不够，还必须“符合条件”
语义相关性只能回答：**“这个内容和问题像不像、有没有关系？”**
但真实系统经常还有明确条件：
> 只看 2026 年的资料；只看某个部门；只看当前版本；只看用户有权限访问的内容。
这就是 🧩 Metadata Filtering 解决的问题。
Metadata Filter 不是在重新判断语义相关性，而是在增加**结构化的硬约束**。
所以可以把两者分开理解：
> Retrieval：**“哪些东西可能与问题有关？”**
>
> Metadata Filtering：**“哪些东西即使相关，也不允许进入结果？”**
具体实现可以是 pre-filter、post-filter，或者在检索过程中直接约束，因此不要把它死记成某个永远固定的位置。更重要的是理解它会改变候选集合：Filter 过严或 Metadata 错误，都可能直接损伤 Recall。
---
# 六、Reranking：候选已经找出来了，现在才值得花更多计算量判断“谁最好”
粗召回以后，候选已经从海量知识缩小到了一个可处理的小集合，但其中仍然可能有噪声。
于是出现 🏆 Reranking。
整个思想是一个非常典型的两阶段检索：
**海量知识 → Retriever 粗召回 → 小候选集合 → Reranker 精排**
Retriever 和 Reranker 的职责不同：
> **Retriever：尽量别漏。**
>
> **Reranker：候选已经在手里了，现在尽量排得准。**
Reranker 可以使用更精细的 Query + Chunk 判断，因此通常比第一阶段更昂贵。如果让它扫描整个知识库，成本太高，所以先粗召回，再只对有限候选精排。
最关键的一条边界再次强调：**Reranker 只能重新排列已经召回的候选，不能召回一个第一阶段完全没找到的 Chunk。**
---
# 七、Context Assembly：找到“好的资料”不等于已经准备好给 LLM
Reranking 得到的是一组更值得使用的候选，但它们仍可能重复、顺序混乱、过长，或者超过当前模型适合使用的 Context Budget。
所以需要 。
它负责：
- 去重
- 排序
- 截取/截断
- 拼接不同来源的内容
- 控制 Context Budget
因此要把它和 Reranking 分开：
> **Reranking 决定“哪些候选更值得留下”；Context Assembly 决定“留下以后怎样组织成 LLM 真正看到的 Context”。**
更多 Context 也不一定更好。上下文越长，成本越高；噪声越多，也越可能把真正重要的信息淹没。
---
# 八、LLM：终于轮到它利用外部知识生成答案
最终交给 LLM 的不是“整个知识库”，而是经过前面一系列筛选后的：
**Query + Context**
LLM 的任务是理解问题、读取提供的外部资料，并据此组织回答。
这也是 RAG 和“让 LLM 自己凭记忆回答”的根本区别：**知识获取被单独拆成了一条可更新、可检索、可控制的外部链路。**
---
# 九、知识更新：离线建库其实是一个持续运行的过程
RAG 知识库不是建完一次就结束。现实中的 Document 会新增、修改、删除。
因此更新过程通常可以理解为：
**Document Change → 找到受影响 Chunk → 重新 Embedding → 更新 Vector DB / Index**
 记录的就是这一部分。
这里有一个很重要的区分：
> **知识变了，不代表 LLM 需要重新训练。**
RAG 的价值之一，就是让外部知识与模型参数解耦。普通业务资料更新通常可以只更新受影响的数据；如果 Embedding 模型本身发生变化，则因为向量空间发生变化，通常需要重新 Embedding 大量已有知识。
---
# 🔗 把整套 RAG 重新串起来
## 离线：把知识变成可检索的数据
**Document**
→ ✂️ Chunking
→（可选 🪄 Contextual Retrieval）
→ 🧭 Embedding
→ Vector DB + Index
其中：
**Chunk Size 的矛盾 → 🧱 Parent-Child Retrieval**
## 在线：从问题找到知识
**Query**
→（可选 🔄 Query Transformation）
→ Retrieval
Retrieval 内部存在不同路线：
**Vector Search** ↔ **Keyword Search**
以及将两者组合起来的：
**🔀 Hybrid Search / BM25**
之后再根据系统设计加入：
**🧩 Metadata Filtering**
→ 🏆 Reranking
→ 
→ LLM
---
# 🤖 RAG 和 Agent 的关系
RAG 不等于 Agent。
可以把它们理解成不同层次：
**Agent：决定“现在需不需要获取外部知识、什么时候获取、获取之后下一步做什么”。**
**RAG：负责“如果要获取知识，怎样从外部知识库里找到有用的信息”。**
**LLM：负责理解、决策和最终生成。**
因此 RAG 可以作为 Agent 的一个 Tool / Capability：
**Agent → 判断需要知识 → 调用 RAG → 获得 Context → 继续决策 / 生成**
RAG 本身仍然是一条独立的知识获取链。
---
# 🧠 最后真正要记住的不是名词，而是这些因果关系
1. **Document 太大、检索粒度太粗 → Chunking。**
2. **Chunk 太小会丢上下文、太大会混入噪声 → Chunk Size 的权衡 → Parent-Child。**
3. **Chunk 脱离原文后缺背景 → Contextual Retrieval。**
4. **文本不好直接做语义距离比较 → Embedding。**
5. **向量很多，逐个比较太慢 → ANN / HNSW。**
6. **语义搜索和精确词项搜索各有盲区 → Vector + Keyword → Hybrid Search。**
7. **海量知识不能一次精确判断 → Retriever 先粗召回，并优先保证 Recall。**
8. **粗召回里仍有噪声 → Reranker 精排；但它救不回没召回的知识。**
9. **相关性之外还有时间、版本、权限等硬条件 → Metadata Filtering。**
10. **检索结果不是天然适合 LLM 的 Context → Context Assembly。**
11. **知识变化不等于模型重新训练 → 更新受影响数据即可。**
**所以 RAG 最终不是“十几个概念排成一条流水线”，而是：一条离线建库主干 + 一条在线检索主干 + Retrieval 内部的不同路线 + 从具体矛盾中长出来的增强策略。**
✂️ Chunking
🧭 Embedding
🗄️ Vector DB
🕸️ HNSW / ANN
🔄 Query Transformation
🔎 Retrieval / Retriever
🧩 Metadata Filtering
🏆 Reranking
🧱 Parent-Child Retrieval
🪄 Contextual Retrieval
🔀 Hybrid Search / BM25
🧺 Context Assembly
🔄 RAG 知识更新