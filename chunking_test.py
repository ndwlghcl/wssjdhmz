from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


def _is_noise(text: str) -> bool:
    """判断是否为"垃圾块"：这类块没有实质内容，却因为高频词（RAG、AI Agent、路径、URL）
    在向量空间里和很多问题都"靠得近"，会抢占检索前排，把真正有用的内容挤下去。"""
    t = text.strip()
    if len(t) < 30:              # 太短：没有信息量
        return True
    if t.startswith("> 路径"):   # 文件头的 "> 路径：xxx / https://..." 引用块
        return True
    return False


def chunking() -> list:
    """读取 data/ 下所有 md → 按标题切 → 超长的递归切 → 过滤垃圾块"""
    BASE_DIR = Path(__file__).parent        # 相对 py 文件定位，不依赖运行目录
    folder = BASE_DIR / "data"

    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
    )
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", " ", ""],   # 中文场景：优先在句号处切断
    )

    all_chunks = []
    dropped = 0
    for path in sorted(folder.glob("*.md")):        # sorted：保证每次建索引的顺序一致
        text = path.read_text(encoding="utf-8")
        docs = md_splitter.split_text(text)
        for d in docs:
            d.metadata["source"] = path.name        # 切之前打标签，子块才能继承到
        for chunk in char_splitter.split_documents(docs):
            if _is_noise(chunk.page_content):
                dropped += 1
                continue
            all_chunks.append(chunk)

    print(f"切块完成：保留 {len(all_chunks)} 块，过滤掉 {dropped} 个垃圾块")
    return all_chunks


if __name__ == "__main__":
    chunks = chunking()
    print(f"\n总块数: {len(chunks)}")
    for c in chunks[:3]:
        print("=" * 60)
        print("metadata:", c.metadata)
        print("content :", c.page_content[:80])
