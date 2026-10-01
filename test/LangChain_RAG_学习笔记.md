# LangChain 与 RAG 学习笔记

> 本笔记整理自一次完整的学习对话，涵盖 LangChain 生态、文本切分原理、RAG 串联、ReAct Agent 健壮性改进与工具重试机制。
> 相关代码位于 `D:\code\my-agent\`（chunking_test.py / RAG_tool.py / ReAct_agent.py / tool_executor.py / tool.py）。

---

## 一、LangChain 生态包的区别

LangChain 0.1 之后做了模块化拆分，各包职责不同：

| 包名 | 定位 | 依赖 | 类比 |
|------|------|------|------|
| `langchain-core` | 基础抽象接口（Runnable、BaseChatModel、消息类型、Prompt） | 被所有包依赖 | 地基 |
| `langchain` | 框架主体（chains、agents 编排） | 依赖 core | 房屋主体 |
| `langchain-community` | 第三方集成大合集（向量库、加载器等） | 依赖 core | 通用配件仓库 |
| `langchain-openai` | OpenAI 专属接入（ChatOpenAI、OpenAIEmbeddings） | 依赖 core | 品牌专用配件 |

**选型原则**：`core` 定规则，`langchain` 搭框架，集成**优先用专属包**（如 `langchain-openai`），实在没有再用 `community`。`langchain-core` 会作为依赖自动安装。

常用安装：

```powershell
pip install langchain langchain-openai
# 需要向量库/加载器再加：
pip install langchain-community
# 国内镜像加速：
pip install langchain langgraph -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 二、"用 OpenAI" 的两种含义

容易混淆，务必分清：

1. **用 OpenAI 的模型**（官方）：调 GPT-4o 等，请求发到 `https://api.openai.com/v1`，用官方 key、官方计费。
2. **用 OpenAI 的接口标准**（OpenAI-Compatible）：别的厂商让服务"长得跟 OpenAI 一样"，于是可以用 `ChatOpenAI` 这个类去调，只要改 `base_url` + `api_key`。

```python
from langchain_openai import ChatOpenAI
# 走 DeepSeek 的 OpenAI 兼容接口：类还是 ChatOpenAI，模型却是 DeepSeek 的
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key="sk-你的DeepSeek的key",
    base_url="https://api.deepseek.com/v1",   # 关键：改地址
)
```

**结论**：`langchain-openai` 提供的是"OpenAI 标准客户端"。连官方就是官方模型，改个 `base_url` 就能连任何兼容服务（DeepSeek、通义千问、智谱、本地 Ollama 等）。

常见兼容地址：
- DeepSeek：`https://api.deepseek.com/v1`
- 通义千问：`https://dashscope.aliyuncs.com/compatible-mode/v1`
- 智谱 GLM：`https://open.bigmodel.cn/api/paas/v4`
- 本地 Ollama：`http://localhost:11434/v1`

---

## 三、langchain-text-splitters 文本切分

安装：`pip install langchain-text-splitters`

### 3.1 Document 对象：正文 + 标签

```python
from langchain_core.documents import Document
doc = Document(
    page_content="正文内容……",                       # 袋子里的纸
    metadata={"source": "01_Chunking.md", "h1": "Chunking"}  # 袋子外的标签
)
```

**一句话**：`Document` = `page_content`（正文）+ `metadata`（附加信息字典）。字符串只有正文，记不住来源/章节，Document 能。

### 3.2 RecursiveCharacterTextSplitter（通用首选）

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    separators=["\n\n", "\n", "。", " ", ""],
)
chunks = splitter.split_text(text)   # list[str]
```

**"递归"的真正含义**：按 `separators` 优先级**从粗到细**尝试——先用 `\n\n` 切，若某块仍超过 `chunk_size`，就对它降级用 `\n` 切，再超就用 `。`、空格，最后用 `""` 逐字符硬切。目的是**尽量在自然边界断开**。

核心参数：

| 参数 | 含义 |
|------|------|
| `chunk_size` | 每块最大长度（默认按字符） |
| `chunk_overlap` | 相邻块重叠量，一般设为 chunk_size 的 10%~20% |
| `separators` | 切分优先级；中文要手动加 `。！？；，` |
| `length_function` | 长度计算函数，默认 `len` |

### 3.3 chunk_overlap 的真相：为什么不是所有块都重叠 ⭐

**实验结论**（chunk_size=100，同一份中文 Markdown）：

- `chunk_overlap=20`：19 个边界里只有 **3 个**真正重叠，重叠字符数 `[0,0,0,0,19,0,0,0,16,0,...,3]`，平均仅 2.0
- `chunk_overlap=60`：**7 个**边界重叠，出现 `53、50、46` 这种整句重叠，平均 10.0

**原因**：overlap 保留的是**完整的分隔单元（句子/行）**，且这些单元总长必须 **≤ chunk_overlap 预算**。
- 边界处是**短标题/短列表项**（≤20 字符）→ 塞得进预算 → 重叠
- 边界处是**长中文句子**（>20 字符）→ 装不下 → 干脆一个不留 → **不重叠**

**教训**：中文一句话普遍 20~40 字符，`chunk_overlap=20` 经常连一句都保留不了，导致重叠**时有时无**。应调大到 **50~80**，让每个边界都能稳定留住一句话。

### 3.4 MarkdownHeaderTextSplitter（结构化切分）

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter
md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "h1"), ("##", "h2")])
docs = md_splitter.split_text(text)   # list[Document]
```

**`("#", "h1")` 元组含义**（易误解）：
- `"#"`：**标记**——去文件里找以 `# ` 开头的行
- `"h1"`：**键名**——把标题文字存进 metadata 时用的 key（可自定义，如 `("##", "章节")`）
- **值**（如 `"Chunking"`）不写在代码里，是从文件对应行 `#` 后面**自动抄来的**

**切分规则**：每遇到一个标题行就开一个新块，两个标题之间的内容成为该块的 `page_content`，标题进 metadata。一个标题的管辖范围 = 从它自己到**下一个同级/更高级标题**之前（所以最后一个 `##` 会一直管到文件结尾）。

**关键细节**：
- `#` 后面**必须有空格**才算标题（遵守 Markdown 标准）。`#Chunking`（无空格）**不被识别**，会被当普通正文原样留在 content，metadata 为 `{}`。
- 被识别的标题行会从 `page_content` 中**移除**（升级为 metadata）。
- 只认标题标记，**不理会逗号/句号/空格**——那是 RecursiveCharacterTextSplitter 的规则。

### 3.5 两步组合切分（处理 Markdown 的标准范式）

MarkdownHeaderTextSplitter 只按标题切、**不管块多大**（一个章节可能几千字）。所以要先按标题切、再按字符切小：

```python
md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#","h1"),("##","h2"),("###","h3")])
docs = md_splitter.split_text(text)          # 带章节 metadata，但块大小不均
char_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50,
                                               separators=["\n\n","\n","。"," ",""])
chunks = char_splitter.split_documents(docs) # 切小，且 metadata 自动继承
```

**必须用 `split_documents` 而非 `split_text`**：前者接收 Document 列表、会把原 metadata 复制到每个子块；后者接收字符串、会丢掉 metadata。

### 3.6 三个方法对比 + split_text 返回类型因类而异 ⭐

| 方法 | 输入 | 输出 |
|------|------|------|
| `split_text` | 字符串 | 见下（因类而异） |
| `create_documents` | 字符串列表 | `list[Document]` |
| `split_documents` | Document 列表 | `list[Document]`（继承 metadata） |

**重要陷阱**：`split_text` 的返回类型**取决于哪个切分器**——
- `RecursiveCharacterTextSplitter.split_text` → `list[str]`（无附加信息，字符串够用）
- `MarkdownHeaderTextSplitter.split_text` → `list[Document]`（标题信息必须装进 metadata，字符串装不下）

同名方法 ≠ 同种返回，Python 里返回什么由**每个类自己的实现**决定。

---

## 四、把切分接入 RAG：Document vs str

### 4.1 类型不匹配的坑

`chunking()` 用 `split_documents` 返回的是 `list[Document]`，但 embedding 接口的 `input` **只接受 str / list[str]**。直接把 Document 列表传进去会报错"模型需要 str 而不是 document"。

**修复**：调用 embedding 前取出正文 `d.page_content`。

### 4.2 metadata 怎么传给模型 ⭐

**核心认知：模型（embedding 模型和大模型）从来只吃字符串，绝不吃 Document 对象或 metadata 字典。**

"把 metadata 传给模型" = **先把 metadata 拼成文字，塞进要发送的字符串里**。数据流中 metadata 有两处会"掉"，都要主动带上：

```python
# build_store：存向量时把 metadata 一起存
return [{"text": d.page_content, "vector": item.embedding, "metadata": d.metadata}, ...]

# retrieval：排序时保留 metadata，最后拼进返回字符串
return "\n\n".join(
    f"[来源 {item['metadata'].get('source','')} | 章节 {item['metadata'].get('h1','')} > "
    f"{item['metadata'].get('h2','')}] [相关度 {item['score']:.2f}]\n{item['text']}"
    for item in selected
)
```

**区分两种用途**：
- 给 **embedding 算向量**：通常只喂正文 `page_content`，不拼 metadata（避免干扰语义相似度）
- 给 **大模型看**（retrieval 返回）：要拼 metadata，让模型知道来源、便于引用和判断可信度

### 4.3 text-embedding-v3 批量上限与分批 ⭐

阿里 DashScope `text-embedding-v3/v4` **单次请求最多 10 条**（官方文档；较早资料为 20），每条 ≤8192 token。超了直接返回 `400 InvalidParameter: batch size is invalid`。RAG 切块常超过 10 条，必须分批：

```python
def build_store(documents: list, batch_size: int = 10) -> list:
    store = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        texts = [d.page_content for d in batch]
        resp = llm.client.embeddings.create(model="text-embedding-v3", input=texts)
        for d, item in zip(batch, resp.data):   # zip 保证顺序对齐，不错位
            store.append({"text": d.page_content, "vector": item.embedding, "metadata": d.metadata})
    return store
```

其他限制：v3 限流 RPM 1800、TPM 120 万。若用 LangChain `OpenAIEmbeddings` 走兼容接口，可设 `chunk_size=10` 自动分批。

---

## 五、ReAct Agent 的 5 项健壮性改进（A~E）

| 优先级 | 问题 | 文件 | 修复 |
|--------|------|------|------|
| 🔴 A | 最终答案未标注来源（metadata 传到了模型却没输出） | ReAct_agent.py | SYSTEM_PROMPT 加规则："结论后标注（来源：文件名 > 章节名）" |
| 🔴 B | 工具执行异常未兜底 | 见第六节（实际在 tool_executor.py） | 捕获异常转成观察结果喂回模型，不崩溃 |
| 🟠 C | 向量未持久化，每次启动重算 embedding | RAG_tool.py | 存盘到 `.embedding_cache.pkl`，用文档指纹判断是否失效 |
| 🟠 D | import 时就建索引，耦合重、失败即崩 | RAG_tool.py | 改惰性初始化 `get_store()`，第一次用时才建 |
| 🟡 E | 历史存超长检索结果，多轮后 token 膨胀 | ReAct_agent.py | `MAX_OBS_CHARS` 截断存入历史的观察结果 |

**C+D 惰性初始化 + 磁盘缓存核心代码**：

```python
_STORE = None
def get_store() -> list:
    global _STORE
    if _STORE is not None:
        return _STORE
    fingerprint = _docs_fingerprint(DOCUMENTS)   # 正文+metadata 算 sha256
    if CACHE_FILE.exists():
        cached = pickle.load(open(CACHE_FILE, "rb"))
        if cached.get("fingerprint") == fingerprint:
            _STORE = cached["store"]              # 命中缓存，不调 API
            return _STORE
    _STORE = build_store(DOCUMENTS)               # 重建
    pickle.dump({"fingerprint": fingerprint, "store": _STORE}, open(CACHE_FILE, "wb"))
    return _STORE
```

效果：首次运行建索引并存盘；再运行直接加载（不调 embedding）；改了笔记内容则指纹变化、自动重建。缓存文件应加进 `.gitignore`。

---

## 六、工具调用的重试机制 ⭐⭐（本次最重要的收获）

### 6.1 重试必须放在"异常还活着"的地方

**关键教训**：异常一旦被某层 `try/except` 转成字符串，上层就再也 `except` 不到它了。

本项目 `ToolExecutor.execute_tool` 早就用 `try/except` 把异常吞成字符串 `"工具执行出错: {e}"` 返回、不再抛出。因此：
- 在 Agent 层（`ReAct_agent.py`）对 `execute_tool` 包 `try/except` 是**死代码**——它根本不抛异常
- **重试只能放进 `execute_tool` 内部**，在 `except` 到原始 openai 异常的那一刻处理

调用链验证：`retrieval` → `embed` → `llm.client.embeddings.create` 抛的 openai 异常，经 `FunctionTool.run`（不吞异常）向上传到 `execute_tool` 的 try——所以那里是唯一能拿到原始异常的位置。

### 6.2 按异常类型精确判断（openai 3.16.2）

```python
import openai
TRANSIENT_EXCEPTIONS = (          # 临时性错误：重试有意义
    openai.APITimeoutError,       # 超时（是 APIConnectionError 的子类）
    openai.APIConnectionError,    # 网络连接问题
    openai.RateLimitError,        # 429 限流
    openai.InternalServerError,   # 5xx 服务端临时故障
)
```

- **临时性错误**（上面 4 种）→ 重试可能自愈
- **永久性错误**（`BadRequestError` 400、`AuthenticationError` 401、`NotFoundError` 404 等）→ 落到通用 `except Exception`，**立即返回不重试**（参数错/密钥错，重试 100 次也没用）

比"关键词匹配错误信息"更精确：不会因消息里恰好含 "timeout" 而误判。

### 6.3 指数退避（exponential backoff）

```python
def execute_tool(self, name: str, **kwargs):
    tool = self.tools.get(name)
    if tool is None:
        return f"错误: 不存在名为 {name} 的工具"
    attempt = 0
    while True:
        try:
            return tool.run(**kwargs)
        except TRANSIENT_EXCEPTIONS as e:
            if attempt >= TOOL_MAX_RETRIES:               # 重试用尽
                return f"工具执行出错（已重试 {attempt} 次仍未成功）: {type(e).__name__}: {e}"
            attempt += 1
            delay = TOOL_RETRY_BASE_DELAY * (2 ** (attempt - 1))   # 1s→2s→4s
            time.sleep(delay)
        except Exception as e:                            # 永久错误
            return f"工具执行出错: {type(e).__name__}: {e}"
```

退避原因：网络/服务端临时过载时立刻重试很可能再失败，逐次拉长间隔给对方恢复时间。代价：`MAX_RETRIES=3` + 基准 1s 时，最坏一次调用卡 1+2+4=7s 才放弃（可按"响应速度 vs 成功率"取舍调参）。

### 6.4 三场景验证结果（retry_demo.py 实测）

| 场景 | 异常 | 行为 | 证明 |
|------|------|------|------|
| 1 | APITimeoutError（第3次成功） | 退避重试2次后成功 | 网络波动能自愈，不惊动模型 |
| 2 | BadRequestError（400） | 立即返回，0 重试 | 不该重试的绝不浪费 |
| 3 | APITimeoutError（一直失败） | 重试3次用尽后放弃 | 有上限，不会卡死 |

三层递进关系：**能自己解决的不麻烦模型（重试）→ 解决不了的如实上报（B）→ 上报内容还要控制体积（E）**。

---

## 七、工程经验小结

1. **验证代码本身也会错**：第一版 overlap 检测用 `prev[-40:]` 窗口，漏判了 overlap=60 的真实重叠区，得出"20 和 60 没区别"的错误结论。修正为"计算 prev 后缀与 curr 前缀的最长公共重合"后真相才浮现。**当实验结果与预期矛盾时，先怀疑测量方法。**

2. **PowerShell 里别跑含 Python 语法的内联命令**：`python -c "..."` 里的逗号、括号、方括号会被 PowerShell 当成自己的语法解析而报错。替代方案：把代码写进 `.py` 临时文件再执行。

3. **看清被调函数的契约再写兜底**：`execute_tool` 契约是"永不抛异常、只返回字符串"，在它外面再包 try/except 就是无效防御。

4. **模型只吃字符串**：任何结构化信息（metadata、Document）想给模型，都必须先序列化成文字。

5. **善用实验验证假设**：本次多个结论（overlap 行为、`#` 无空格不识别、重试逻辑）都是"提出假设 → 写代码跑 → 用输出印证"得到的，比死记硬背扎实得多。

---

## 附：本项目验证用脚本（位于工作区 D:\code\hello-agents\）

- `chunking_overlap_compare.py`：验证 chunk_overlap 大小对"重叠是否稳定生效"的影响
- `md_header_test.py`：验证 MarkdownHeaderTextSplitter 对 `# 标题`（有空格）与 `#标题`（无空格）的处理差异
- `retry_demo.py`：用真实 openai 异常演示按类型重试的三种场景
