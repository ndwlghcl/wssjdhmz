"""检索调试：打印某个问题在检索各阶段的排名，用来定位"为什么排成这个顺序"。

用法：python test/inspect_query.py "为什么需要 Reranking？"
（不传问题时用默认问题）
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from RAG_tool import (
    RECALL_N,
    RECALL_THRESHOLD,
    RERANK_THRESHOLD,
    embed,
    get_index,
    llm,
    score_all,
)


def main():
    question = sys.argv[1] if len(sys.argv) > 1 else "为什么需要 Reranking？"
    store = get_index()["store"]

    q = embed(question)
    scores = score_all(q)
    order = np.argsort(-scores)

    print(f"问题: {question}")
    print(f"\n【阶段一】余弦排序（RECALL_N={RECALL_N}，阈值 {RECALL_THRESHOLD}）")
    cands = []
    for i in order[:RECALL_N]:         # 必须和生产代码一样限制条数，否则看到的不是真实候选集
        s = float(scores[i])
        if s < RECALL_THRESHOLD:
            break                      # 降序排列，一旦低于阈值后面都低于
        cands.append({
            "text": store[i]["text"],
            "metadata": store[i]["metadata"],
            "score": s,
        })
    for pos, c in enumerate(cands[:10], 1):
        m = c["metadata"]
        print(f"  {pos:2d}. cosine {c['score']:.3f} | {m.get('source', '')} > {m.get('h2', '')}")

    print(f"\n送去 rerank 的候选数: {len(cands)}")
    print(f"\n【阶段二】rerank 精排（阈值 {RERANK_THRESHOLD}）")
    results = llm.rerank(
        query=question,
        documents=[c["text"] for c in cands],
        top_n=min(10, len(cands)),
    )
    for r in results:
        c = cands[r["index"]]
        m = c["metadata"]
        flag = "✅" if r["relevance_score"] >= RERANK_THRESHOLD else "⛔ 被阈值拦下"
        print(f"  {flag} rerank {r['relevance_score']:.3f} | cosine {c['score']:.3f} "
              f"| {m.get('source', '')} > {m.get('h2', '')}")
        print(f"        {c['text'][:60]}...")


if __name__ == "__main__":
    main()
