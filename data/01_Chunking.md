# ✂️ Chunking

> 路径：AI Agent / 📚 RAG
>
> https://app.notion.com/p/3d1776baec498143ad18c9dac81b49b1?pvs=204

Chunking 不是简单地把长文本切成几段，而是在定义 **Retrieval 的基本单位**。
原始 Document 往往很长。如果整篇文档作为一个检索单位，局部问题很难精准定位；如果切得过碎，又会让原本依赖上下文才能理解的知识失去背景。因此 Chunking 从一开始就在解决一个核心矛盾：
> **检索粒度越细，越容易精准定位；上下文越完整，越容易正确理解。两者通常不能同时无限增加。**
## 1. 为什么必须切 Chunk？
Embedding、Retrieval 最终都需要一个“可以被表示、比较、返回”的知识单位。整篇文档通常太粗：用户问其中一句，系统却只能把整个文档作为候选。
所以 Chunking 做的第一件事，是把 Document 转换成一组可独立参与检索的知识单元。
这也解释了为什么 Chunking 会直接影响后面的 Retrieval：**你怎么切，实际上就在决定 Retriever 能以多细的粒度找到知识。**
## 2. Chunk 太小和太大的代价
**太小：**
- 检索粒度细，容易命中局部信息
- Embedding 更聚焦
- 但指代、条件、前因后果容易被切断
- 单独拿出来可能“不知道它在说谁”
**太大：**
- 上下文更完整
- 但语义被大量无关内容稀释
- 一个 Chunk 内可能同时包含很多主题
- 检索和后续 LLM 上下文成本更高
所以没有一个脱离场景的“最佳 Chunk Size”。真正需要平衡的是 **检索精度、上下文完整性、Embedding/检索成本、LLM Context Cost**。
## 3. 为什么需要 Overlap？
如果直接在边界切割，恰好跨越边界的一句话可能被拆开。Overlap 让相邻 Chunk 保留一部分共同内容，从而降低边界切割造成的信息断裂。
它解决的是“切割边界”的问题，而不是“Chunk 应该多大”的问题。Overlap 太大也会造成重复内容和存储、检索成本增加。
## 4. Chunking 的常见策略
- **Fixed-size**：按字符/token 数量切。简单、稳定，但不理解文档结构。
- **Recursive**：优先在段落、句子等较自然边界切分，再逐步细化，是常见的通用方案。
- **Structure-aware**：利用标题、章节、列表、表格等文档结构，让 Chunk 更符合原始语义组织。
- **Semantic Chunking**：根据语义变化寻找边界，目标是让一个 Chunk 内的内容更具有语义一致性，但实现成本更高。
这些不是“谁绝对更先进”，而是在不同数据上解决不同的切分问题。
## 5. Chunking 与后续概念的关系
当 Chunk 太小会丢上下文、太大又影响定位时，就自然出现了 **Parent-Child Retrieval**：让较小的 Child 负责定位，让较大的 Parent 负责提供上下文。
当 Chunk 即使大小合适，单独拿出来仍然缺少背景时，则可以使用 **Contextual Retrieval**，在 Embedding 前为 Chunk 补充必要上下文。
所以：
> **Parent-Child 解决“检索大小和返回大小不必相同”；Contextual Retrieval 解决“Chunk 本身缺背景”。**