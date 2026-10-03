import hashlib
import pickle
from pathlib import Path
from typing import Annotated

from chunking_test import chunking
from config import Config
from llm import LLM
from tool import tool


# 只创建一个 LLM 客户端，全局复用
config = Config.from_env()
llm = LLM(config=config)

# 向量索引的磁盘缓存文件（和脚本同目录）
CACHE_FILE = Path(__file__).parent / ".embedding_cache.pkl"

# 读取并切分文档（只读文件+切分，不调用 API，放在 import 阶段没问题）
DOCUMENTS = chunking()


def cosine(a: list, b: list) -> float:
    """余弦相似度：衡量两个向量方向上的接近程度"""
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
        for d, item in zip(batch, resp.data):
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
            with open(CACHE_FILE, "rb") as f:
                cached = pickle.load(f)
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
        with open(CACHE_FILE, "wb") as f:
            pickle.dump({"fingerprint": fingerprint, "store": _STORE}, f)
        print("💾 向量索引已保存到磁盘缓存")
    except Exception as e:
        print(f"⚠️ 缓存写入失败（不影响本次使用）：{e}")
    return _STORE


# 相似度阈值：低于这个分数视为"不相关"，宁可不返回也不给噪声
# 实测分布：相关问题 ≈0.80，不相关问题 ≈0.42，所以取中间的 0.5
SIM_THRESHOLD = 0.5


@tool(name="retrieval", description="根据问题检索本地知识库。输入应该是一个问题，返回最相关的资料片段")
def retrieval(
    input: Annotated[str, "要检索的问题，必须是一句完整的问句，不要只给关键词"],
    top_k: Annotated[int, "返回的资料片段数量，默认 4，一般无需修改"] = 4,
) -> str:
    store = get_store()               # D：用到时才确保索引已就绪
    q_embedding = embed(input)

    score = [
        {"text": item["text"], "metadata": item["metadata"],   # 带上 metadata
         "score": cosine(q_embedding, item["vector"])}
        for item in store
    ]

    ranking = sorted(score, key=lambda item: item["score"], reverse=True)
    selected = [item for item in ranking[:top_k] if item["score"] >= SIM_THRESHOLD]
    if not selected:
        return "知识库中没有找到与问题相关的内容。"

    # 把 metadata（来源、章节）拼进文字，模型才"看得到"
    return "\n\n".join(
        f"[来源 {item['metadata'].get('source', '')} | "
        f"章节 {item['metadata'].get('h1', '')} > {item['metadata'].get('h2', '')}] "
        f"[相关度 {item['score']:.2f}]\n{item['text']}"
        for item in selected
    )


if __name__ == "__main__":
    store = get_store()

    # 校准用：每个问题只打印相似度最高的前 5 块，
    # 观察"相关问题"和"无关问题"的分数分布，用来定 SIM_THRESHOLD
    questions = [
        "Chunking 有哪些策略？",      # RAG 笔记（应命中 01_Chunking.md）
        "俄耳甫斯为什么回头？",        # 思想笔记（应命中 38_...md）
        "梯度下降是什么？",            # 数学笔记（应命中 23_...md）
        "草莓蛋糕怎么做？",            # 无关问题（应该全部低分）
    ]
    for q in questions:
        q_emb = embed(q)
        scored = sorted(
            (
                {
                    "score": cosine(q_emb, item["vector"]),
                    "meta": item["metadata"],
                    "text": item["text"],
                }
                for item in store
            ),
            key=lambda x: x["score"],
            reverse=True,
        )
        print(f"\n问题: {q}")
        for item in scored[:5]:
            src = item["meta"].get("source", "")
            h2 = item["meta"].get("h2", "")
            print(f"  {item['score']:.3f}  {src} > {h2}  | {item['text'][:20]}...")

    print("\n--- 工具实际返回 ---")
    # 注意：装饰后的 retrieval 是 FunctionTool 对象，不是函数，要用 .run() 调用
    print(retrieval.run(input="品味有什么用？"))