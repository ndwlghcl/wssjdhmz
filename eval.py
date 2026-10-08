"""检索回归测试（Eval）：跑一组固定问题，检查是否命中期望的文件。

用法：python eval.py

为什么只测检索、不测最终回答：
- 回答由模型生成，同一问题两次可能措辞不同 → 不适合做"通过/失败"判定
- 检索命中哪个文件是确定性的（同一批参数下结果可复现）→ 适合做回归
所以每次改动检索相关代码/参数（切块、阈值、RECALL_N、rerank）后，跑一遍这个脚本，
看命中率有没有退化。

判定标准：
- EXPECT 有值 → 该文件必须出现在返回结果的 [来源 xxx] 里（命中）
- EXPECT 为 None → 必须返回"知识库中没有找到与问题相关的内容。"（拒答）
"""
import re

from RAG_tool import retrieval

# (问题, 期望命中的文件 / None 表示期望拒答)
# 期望值可以是 str（单个文件）或 tuple（多个可接受文件，任一命中即可）
CASES = [
    # 原有验收问题
    ("Chunking 有哪些策略？", "01_Chunking.md"),
    ("俄耳甫斯为什么回头？", "38_俄耳甫斯与欧律狄刻爱是否让回头成为必然.md"),
    ("梯度下降是什么？", "23_梯度下降与极值.md"),
    ("品味有什么作用", "36_品味与创造不要过早确信自己已经知道.md"),
    ("草莓蛋糕怎么做？", None),
    # 新增：覆盖不同主题的文件
    ("反向传播是怎么工作的？", "30_反向传播.md"),
    ("Transformer 的整体结构是什么？", "22_Transformer_总体结构.md"),
    # 经人工确认：00_RAG.md 里"粗召回后仍有噪声，于是出现 Reranking"那段
    # 比 06_Reranking.md 的小节更直接回答这个问题，两者都算命中
    ("为什么需要 Reranking？", ("00_RAG.md", "06_Reranking.md")),
    ("红烧肉怎么做？", None),
]

SOURCE_RE = re.compile(r"\[来源 (.+?) \|")


def run_case(question: str, expect) -> tuple[bool, str]:
    """返回 (是否通过, 说明)。expect：None=期望拒答，str/tuple=可接受的来源文件"""
    text = retrieval.run(input=question, top_k=4)

    if expect is None:
        if "没有找到" in text:
            return True, "正确拒答"
        sources = SOURCE_RE.findall(text)
        return False, f"期望拒答，实际返回了 {len(sources)} 条"

    accepted = (expect,) if isinstance(expect, str) else expect
    sources = SOURCE_RE.findall(text)
    if not sources:
        return False, "检索为空（被判为不相关）"

    best = None
    for name in accepted:
        if name in sources:
            rank = sources.index(name) + 1
            if best is None or rank < best[1]:
                best = (name, rank)
    if best:
        return True, f"命中（第 {best[1]} 位）：{best[0]}"
    return False, f"未命中，可接受 {list(accepted)}，实际 {sources}"


def main():
    passed = 0
    first_hit = 0
    total = len(CASES)

    print(f"检索回归测试：{total} 个用例\n" + "-" * 60)
    for question, expect in CASES:
        ok, detail = run_case(question, expect)
        passed += ok
        if ok and expect is not None and "第 1 位" in detail:
            first_hit += 1
        print(f"{'✅' if ok else '❌'} {question}")
        print(f"     {detail}")

    print("-" * 60)
    print(f"命中率: {passed}/{total}")
    print(f"首选命中: {first_hit}/{total - len([c for c in CASES if c[1] is None])}（多选题里排第一的比例）")


if __name__ == "__main__":
    main()
