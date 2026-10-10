vector_results = ["A", "B", "C"]
bm25_results = ["B", "D", "A"]

scores = {}

# 向量结果：权重为 2
for rank, doc_id in enumerate(vector_results, start=1):
    contribution = 2 / (60 + rank)
    scores[doc_id] = scores.get(doc_id, 0) + contribution

# BM25 结果：权重为 1
for rank, doc_id in enumerate(bm25_results, start=1):
    contribution = 1 / (60 + rank)
    scores[doc_id] = scores.get(doc_id, 0) + contribution

# 每项是（文档编号，分数），按分数从大到小排序
ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)

for doc_id, score in ranked:
    print(doc_id, round(score, 6))