from chunking_test import chunking
from llm import LLM
from tool import tool


# 只创建一个 LLM 客户端，全局复用
llm = LLM()

# 知识库原文（后续可以换成从文件读取）
# DOCUMENTS = [
#     "“俄耳甫斯回头”不是‘我选择了看你一眼，哪怕因此失去你’，也不是‘我为了证明爱而回头’。更接近的是：**他想不回头。可是他无法把所爱的人变成一个抽象的、不可见的存在。爱具体的、鲜活的欧律狄刻，所以爱让他回头。**这也是故事最惆怅的地方：**他不是选择了一个错误的选项，而是连不犯这个错误的自由都没有。**",
#     "一个人认为宇宙没有赋予生命意义当然也可以继续体验生命，从来都没有‘应该’，没有应该结束生命，只有想不想去做的区别。",
#     "“品味”让人能够判断一个东西好不好、成熟不成熟、优秀还是平庸。随着阅读、观看和经验增加，一个人的判断标准会越来越丰富，也越来越能够迅速识别某种作品或经验属于什么类型。但这种能力同时存在一个反作用：当一个人已经拥有大量成熟的判断模板时，面对新的东西，可能会下意识地把它迅速归类，用过去已经形成的标准判断它，而不是暂时放下既有结论，真正观察它是什么。",
# ]

DOCUMENTS = chunking()

def cosine(a: list, b: list) -> float:
    """余弦相似度：衡量两个向量方向上的接近程度"""
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x ** 2 for x in a) ** 0.5
    norm_b = sum(x ** 2 for x in b) ** 0.5
    return dot_product / (norm_a * norm_b)


# def build_store(documents: list) -> list:
#     """把文档批量转成向量，和原文配对存起来（只做一次）"""
#     resp = llm.client.embeddings.create(model="text-embedding-v3", input=documents)
#     return [
#         {"text": text, "vector": item.embedding}
#         for text, item in zip(documents, resp.data)
#     ]

# def build_store(documents: list) -> list:
#     """把文档批量转成向量，连同正文和 metadata 一起存起来（只做一次）"""
#     texts = [d.page_content for d in documents]   # embedding 只吃正文
#     resp = llm.client.embeddings.create(model="text-embedding-v3", input=texts)
#     return [
#         {"text": d.page_content, "vector": item.embedding, "metadata": d.metadata}
#         for d, item in zip(documents, resp.data)   # ← 把 metadata 也带上
#     ]

def build_store(documents: list, batch_size: int = 10) -> list:
    """分批把文档转成向量，规避 text-embedding-v3 单次最多 10 条的限制"""
    store = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]              # 每批最多 batch_size 个 Document
        texts = [d.page_content for d in batch]          # Document -> str
        resp = llm.client.embeddings.create(model="text-embedding-v3", input=texts)
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
    resp = llm.client.embeddings.create(model="text-embedding-v3", input=[text])
    return resp.data[0].embedding


# 模块加载时就建好索引，之后每次检索直接复用，不再重复嵌入
STORE = build_store(DOCUMENTS)


# 相似度阈值：低于这个分数视为"不相关"，宁可不返回也不给噪声
# 实测分布：相关问题 ≈0.80，不相关问题 ≈0.42，所以取中间的 0.5
SIM_THRESHOLD = 0.5

@tool(name="retrieval", description="根据问题检索本地知识库。输入应该是一个问题，返回最相关的资料片段")
def retrieval(input: str, top_k: int = 2) -> str:
    q_embedding = embed(input)

    score = [
        {"text": item["text"], "metadata": item["metadata"],   # ← 带上 metadata
         "score": cosine(q_embedding, item["vector"])}
        for item in STORE
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
# @tool(name="retrieval", description="根据问题检索本地知识库。输入应该是一个问题，返回最相关的资料片段")
# def retrieval(input: str, top_k: int = 2) -> str:
#     q_embedding = embed(input)
#
#     score = [
#         {"text": item["text"], "score": cosine(q_embedding, item["vector"])}
#         for item in STORE
#     ]
#
#     # 按 score 从大到小排序，取前 top_k 段
#     ranking = sorted(score, key=lambda item: item["score"], reverse=True)
#
#     # 过滤掉相关性太低的：宁可告诉模型"没找到"，也不塞无关内容
#     selected = [item for item in ranking[:top_k] if item["score"] >= SIM_THRESHOLD]
#     if not selected:
#         return "知识库中没有找到与问题相关的内容。"
#
#     # 带上相关度分数，让模型自己判断该信多少
#     return "\n\n".join(
#         f"[相关度 {item['score']:.2f}] {item['text']}" for item in selected
#     )


if __name__ == "__main__":
    # 校准用：打印问题与每段资料的相似度，用来确定 SIM_THRESHOLD 该设多少
    for q in ["品味有什么用？", "俄耳甫斯为什么回头？", "草莓蛋糕怎么做？"]:
        q_emb = embed(q)
        print(f"\n问题: {q}")
        for item in STORE:
            print(f"  相似度 {cosine(q_emb, item['vector']):.3f}  {item['text'][:18]}...")

    print("\n--- 工具实际返回 ---")
    # 注意：装饰后的 retrieval 是 FunctionTool 对象，不是函数，要用 .run() 调用
    print(retrieval.run(input="品味有什么用？"))
