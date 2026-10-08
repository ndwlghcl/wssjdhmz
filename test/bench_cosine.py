"""对比：矩阵版打分 vs 纯 Python 版逐项打分（同一批数据、同样次数）"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from RAG_tool import cosine, embed, get_index, score_all

idx = get_index()
store = idx["store"]
q = embed("Chunking 有哪些策略？")
N = 50

t = time.perf_counter()
for _ in range(N):
    score_all(q)
matrix_time = time.perf_counter() - t

t = time.perf_counter()
for _ in range(N):
    [cosine(q, item["vector"]) for item in store]
pure_time = time.perf_counter() - t

print(f"库大小: {len(store)} 条")
print(f"矩阵版:   {matrix_time:.3f}s / {N} 次")
print(f"纯Python: {pure_time:.3f}s / {N} 次")
print(f"提速: {pure_time / matrix_time:.1f}x")
