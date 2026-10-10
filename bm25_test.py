# import jieba
# from rank_bm25 import BM25Okapi
#
# documents = [
#     "HNSW通过多层图加速检索",
#     "BM25根据关键词匹配文档",
#     "向量检索使用embedding表示语义",
# ]
#
# # 每条文档都拆成词列表
# document_words = [jieba.lcut(doc) for doc in documents]
#
# # 统计这些文档的词频、长度等信息
# bm25 = BM25Okapi(document_words)
#
# # 问题也拆成词列表，再计算每条文档的分数
# # query_words = jieba.lcut("HNSW为什么快")
# query_words = jieba.lcut("怎样提高查找速度")
# scores = bm25.get_scores(query_words)
#
# for document, score in zip(documents, scores):
#     print(round(float(score), 3), document)

import jieba
from rank_bm25 import BM25Okapi
from RAG_tool import DOCUMENTS, embed, score_all, RECALL_THRESHOLD

chunks = DOCUMENTS

# 文档和问题都转成小写，再分词
document_words = [
    jieba.lcut(chunk.page_content.lower())
    for chunk in chunks
]
bm25 = BM25Okapi(document_words)

query = "HNSW为什么能加速检索"
scores = bm25.get_scores(jieba.lcut(query.lower()))

# 按分数排序，得到的是 chunk 编号
order = sorted(
    range(len(chunks)),
    key=lambda i: scores[i],
    reverse=True,
)

# 保留正分结果，取前五条
bm25_results = [i for i in order if scores[i] > 0][:5]

print("BM25 结果编号：", bm25_results)

for chunk_id in bm25_results:
    chunk = chunks[chunk_id]
    print(chunk_id, round(float(scores[chunk_id]), 3))
    print("来源：", chunk.metadata["source"])
    print(chunk.page_content[:100])
    print()


# 把问题转成向量，再计算与每个片段的相似度
vector_scores = score_all(embed(query))

# 按相似度从高到低排列片段编号
vector_order = sorted(
    range(len(chunks)),
    key=lambda i: vector_scores[i],
    reverse=True,
)

# 沿用你现有的相似度阈值，取前五条
vector_results = [
    i for i in vector_order
    if vector_scores[i] >= RECALL_THRESHOLD
][:5]

print("向量结果编号：", vector_results)
print("BM25 结果编号：", bm25_results)

rrf_scores = {}

for rank, chunk_id in enumerate(vector_results, start=1):
    contribution = 2 / (60 + rank)
    rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + contribution

for rank, chunk_id in enumerate(bm25_results, start=1):
    contribution = 1 / (60 + rank)
    rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + contribution

fused_results = sorted(
    rrf_scores.items(),
    key=lambda item: item[1],
    reverse=True,
)

for chunk_id, score in fused_results:
    print(chunk_id, round(score, 6))

from RAG_tool import llm

# 提取编号，再按这个顺序取出正文
candidate_ids = [chunk_id for chunk_id, score in fused_results]
candidate_texts = [
    chunks[chunk_id].page_content
    for chunk_id in candidate_ids
]

# 调用你已有的 rerank，选出前四条
reranked = llm.rerank(
    query=query,
    documents=candidate_texts,
    top_n=4,
)

for item in reranked:
    # 把候选列表的位置，换回原来的 chunk 编号
    chunk_id = candidate_ids[item["index"]]
    chunk = chunks[chunk_id]

    print("编号：", chunk_id)
    print("rerank 分数：", round(item["relevance_score"], 3))
    print("来源：", chunk.metadata["source"])
    print(chunk.page_content[:150])
    print()