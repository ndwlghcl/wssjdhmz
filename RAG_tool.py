import hashlib
import json
from pathlib import Path
from typing import Annotated

import numpy as np

from chunking_test import chunking
from config import Config
from llm import LLM
from tool import tool
import jieba
from rank_bm25 import BM25Okapi

# 只创建一个 LLM 客户端，全局复用
config = Config.from_env()
llm = LLM(config=config)

# 向量索引的磁盘缓存文件（和脚本同目录）
# 用 JSON 而非 pickle：store 是纯数据（字典/字符串/浮点），明文可调试、
# 且加载是纯解析，缓存文件被篡改也不可能执行代码
CACHE_FILE = Path(__file__).parent / ".embedding_cache.json"

# 读取并切分文档（只读文件+切分，不调用 API，放在 import 阶段没问题）
DOCUMENTS = chunking()


def cosine(a: list, b: list) -> float:
    """纯 Python 版余弦相似度（保留作为对照实现，检索主路径已改用 numpy 矩阵）。
    教学价值：看着这三行，能对照理解下面的矩阵版在做什么"""
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x ** 2 for x in a) ** 0.5
    norm_b = sum(x ** 2 for x in b) ** 0.5
    return dot_product / (norm_a * norm_b)


def build_store(documents: list, batch_size: int = 10) -> list:
    """分批把文档转成向量，规避 text-embedding-v3 单次最多 10 条的限制"""
    store = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]              # 每批最多 batch_size 个 Document
        texts = [d.page_content for d in batch]          # Document -> str
        resp = llm.embed(texts)   # 走 LLM.embed，临时错误自动重试
        # 按顺序把这一批的结果拼回 store，metadata 一起带上
        # for d, item in zip(batch, resp.data):
        #     store.append({
        #         "text": d.page_content,
        #         "vector": item.embedding,
        #         "metadata": d.metadata,
        #     })

        by_index = {item.index: item for item in resp.data}

        if len(by_index) != len(texts):
            # 宁可炸,也不要静默错位——这里炸掉是有价值的失败
            raise RuntimeError(
                f"embedding 返回条数不符:请求 {len(texts)} 条,返回 {len(by_index)} 条"
            )

        for j, d in enumerate(batch):
            item = by_index[j]  # 用 index 显式取,不靠位置
            store.append({
                "text": d.page_content,
                "vector": item.embedding,
                "metadata": d.metadata,
            })
    return store


def embed(text: str) -> list:
    """把一段文字转成向量"""
    resp = llm.embed([text])
    return resp.data[0].embedding


def _docs_fingerprint(documents: list) -> str:
    """根据所有文档的正文 + metadata 算一个指纹。
    文档内容或来源/章节一旦变化，指纹就变，用来判断磁盘缓存是否失效。"""
    h = hashlib.sha256()
    h.update(config.embedding_model.encode("utf-8"))
    for d in documents:
        h.update(d.page_content.encode("utf-8"))
        h.update(repr(sorted(d.metadata.items())).encode("utf-8"))
    return h.hexdigest()


# 惰性缓存：模块加载时不构建，第一次用到才建，之后复用同一份
_STORE = None


def get_store() -> list:
    """惰性初始化 + 磁盘缓存（解决 C、D）：
    - 第一次调用时才构建索引，import 阶段不再触发 embedding 请求
    - 构建结果存盘，下次启动若文档未变则直接加载，不重复花钱调 API
    """
    global _STORE
    if _STORE is not None:            # 本次进程内已建好，直接复用
        return _STORE

    fingerprint = _docs_fingerprint(DOCUMENTS)

    # 1) 先尝试从磁盘加载缓存
    if CACHE_FILE.exists():
        try:
            # encoding 必须显式：Windows 文本模式默认 GBK，中文内容读写都会炸
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("fingerprint") == fingerprint:
                print("✅ 已从磁盘缓存加载向量索引（未重新调用 embedding）")
                _STORE = cached["store"]
                return _STORE
            print("♻️ 检测到文档已变化，重新构建向量索引…")
        except Exception as e:
            print(f"⚠️ 缓存读取失败，将重新构建：{e}")

    # 2) 缓存不可用或已失效 → 重新构建，并存盘
    _STORE = build_store(DOCUMENTS)
    try:
        # ensure_ascii=False：中文保持原样，不转 \uXXXX（否则文件更大更不可读）
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({"fingerprint": fingerprint, "store": _STORE}, f, ensure_ascii=False)
        print("💾 向量索引已保存到磁盘缓存")
    except Exception as e:
        print(f"⚠️ 缓存写入失败（不影响本次使用）：{e}")
    return _STORE


# ===== numpy 加速视图 =====
# store 是唯一数据源（含磁盘缓存）；matrix / norms 只是把它的向量换个形式摆放，
# 用来"一次算完全部分数"。矩阵第 i 行 == store 第 i 项，靠索引对齐，不是另一份存储。
_INDEX = None


def get_index() -> dict:
    """惰性构建一次：store + 向量矩阵 (N, 1024) + 每行范数 (N,) + bm25"""
    global _INDEX
    if _INDEX is not None:
        return _INDEX

    store = get_store()
    matrix = np.array([item["vector"] for item in store])

    document_words = [
        jieba.lcut(item["text"].lower())
        for item in store
    ]

    _INDEX = {
        "store": store,
        "matrix": matrix,
        "norms": np.linalg.norm(matrix, axis=1),  # 预计算每行长度，避免每次重复算
        "bm25": BM25Okapi(document_words),
    }
    return _INDEX


def score_all(query_vector: list) -> "np.ndarray":
    """一次算出 query 与全库所有向量的余弦分数，返回 (N,) 数组。
    对照纯 Python 版：sum(x*y) → 矩阵乘 @；sum(x**2)**0.5 → np.linalg.norm。
    区别在于：这里没有 for 循环，198 次点积合并成一次矩阵运算"""
    idx = get_index()
    q = np.array(query_vector)
    return idx["matrix"] @ q / (idx["norms"] * np.linalg.norm(q))


# ===== 检索参数（均为实测校准值，改语料后需重新校准）=====
RECALL_N = 20              # 每路召回上限：融合去重后最多 40 条交给 rerank
RECALL_THRESHOLD = 0.4     # 粗筛阈值。实测：相关问题余弦 ≥0.589，无关问题 ≤0.260，取中间偏保守
RERANK_THRESHOLD = 0.4     # 精排阈值。实测：相关问题 rerank ≥0.571，无关问题 ≤0.292，取中间偏保守


@tool(name="retrieval", description="根据问题检索本地知识库。输入应该是一个问题，返回最相关的资料片段")
def retrieval(
    input: Annotated[str, "要检索的问题，必须是一句完整的问句，不要只给关键词"],
    top_k: Annotated[int, "返回的资料片段数量，默认 4，一般无需修改"] = 4,
) -> str:
    idx = get_index()      # 两路检索复用同一份索引，片段编号按 store 对齐
    store = idx["store"]
    q_embedding = embed(input)

    # numpy 矩阵版：一次算完全库分数（替代原来"逐项 cosine + sorted"）
    vector_scores = score_all(q_embedding)
    vector_order = np.argsort(-vector_scores)       # argsort 默认升序，取负号变降序
    vector_results = [
        i for i in vector_order
        if vector_scores[i] >= RECALL_THRESHOLD
    ][:RECALL_N]

    bm25 = idx["bm25"]

    bm25_scores = bm25.get_scores(jieba.lcut(input.lower()))

    # 按分数排序，得到的是 chunk 编号
    bm25_order = sorted(
        range(len(store)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )
    bm25_results = [i for i in bm25_order if bm25_scores[i] > 0][:RECALL_N]

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


    # ① 取融合后的候选：每项是 (片段编号, RRF 分数)。
    # 余弦阈值已在向量路使用；此处保留 BM25 单独召回的片段。
    # 保留两路并集：若再截成 20 条，2:1 权重会在向量路满额时排除所有 BM25 独有项。
    candidates = [
        {
            "text": store[chunk_id]["text"],
            "metadata": store[chunk_id]["metadata"],
            "rrf_score": float(rrf_score),
        }
        for chunk_id, rrf_score in fused_results
    ]
    if not candidates:
        return "知识库中没有找到与问题相关的内容。"

    # ② 精排：把这批候选的正文送去 rerank，让它挑出最相关的 top_k 条
    results = llm.rerank(query=input, documents=[c["text"] for c in candidates], top_n=top_k)

    # ③ 用 index 映射回候选：rerank 只返回索引和分数，metadata 在 candidates 里
    picked = [
        (r["relevance_score"], candidates[r["index"]])
        for r in results
        if r["relevance_score"] >= RERANK_THRESHOLD
    ]
    if not picked:
        return "知识库中没有找到与问题相关的内容。"

    # ④ 拼输出：带上来源和分数（现在是 rerank 分数），模型才能引用
    # 章节回退：有些块直接在 h1 之下（不在任何 ## 小节里），此时用 h3 或"引言"顶上，
    # 避免引用信息出现空白
    def _section(m: dict) -> str:
        return m.get("h2") or m.get("h3") or "引言"

    return "\n\n".join(
        f"[来源 {item['metadata'].get('source', '')} | "
        f"章节 {item['metadata'].get('h1', '')} > {_section(item['metadata'])}] "
        f"[相关度 {score:.2f}]\n{item['text']}"
        for score, item in picked
    )


if __name__ == "__main__":
    store = get_store()

    # 校准用，看两段分数分布：
    # ① 余弦分数 —— 用来定粗筛阈值 RECALL_THRESHOLD
    # ② rerank 分数 —— 用来定精排阈值 RERANK_THRESHOLD
    questions = [
        "Chunking 有哪些策略？",      # RAG 笔记（应命中 01_Chunking.md）
        "俄耳甫斯为什么回头？",        # 思想笔记（应命中 38_...md）
        "梯度下降是什么？",            # 数学笔记（应命中 23_...md）
        "草莓蛋糕怎么做？",            # 无关问题（应该全部低分）
    ]
    # 一致性校验：矩阵版与纯 Python 版算出的分数必须一致（重构不能改变行为）
    q0 = embed(questions[0])
    matrix_scores = score_all(q0)
    max_diff = max(
        abs(float(matrix_scores[i]) - cosine(q0, store[i]["vector"]))
        for i in range(len(store))
    )
    print(f"\n一致性校验：矩阵版 vs 纯 Python 版最大差异 = {max_diff:.2e}（应接近 0）")

    for q in questions:
        q_emb = embed(q)
        scores = score_all(q_emb)
        order = np.argsort(-scores)
        scored = [
            {"score": float(scores[i]), "meta": store[i]["metadata"], "text": store[i]["text"]}
            for i in order
        ]
        print(f"\n问题: {q}")
        print("  余弦 top5:")
        for item in scored[:5]:
            src = item["meta"].get("source", "")
            h2 = item["meta"].get("h2", "")
            print(f"    {item['score']:.3f}  {src} > {h2}  | {item['text'][:20]}...")

        # rerank 视角：对粗筛出的候选做精排，看分数分布
        cands = scored[:RECALL_N]
        results = llm.rerank(query=q, documents=[c["text"] for c in cands], top_n=5)
        print("  rerank top5:")
        for r in results[:5]:
            c = cands[r["index"]]
            print(f"    {r['relevance_score']:.3f}  {c['meta'].get('source', '')} > {c['meta'].get('h2', '')}  | {c['text'][:20]}...")

    # 验收：直接看 rerank 版 retrieval 的最终输出
    # 注意：装饰后的 retrieval 是 FunctionTool 对象，不是函数，要用 .run() 调用
    print("\n--- 工具实际返回（rerank 版）---")
    for q in questions:
        print(f"\n问题: {q}")
        print(retrieval.run(input=q))
