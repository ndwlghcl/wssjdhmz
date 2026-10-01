# Agent 核心组件

> 路径：AI Agent
>
> https://app.notion.com/p/3d1776baec49814189a8eb2d589ced69?pvs=204

这一层记录一个 Agent 系统内部反复出现的核心能力。它们不是必须同时存在的“固定组件”，而是 Agent 为了**持续行动、记住信息、处理复杂任务和应对失败**而逐渐引入的能力。
## 🧭 它们怎样组成 Agent
可以先把 Agent 看成一个循环：
`Context → LLM → 决策 → Tool Calling → Tool Runtime → Observation → Context → …`
这个循环解决了“模型只能回答一次”的问题。模型做出行动后拿到新的结果，再基于新结果继续判断，直到任务结束。
在这个基础循环上，再根据任务复杂度加入不同能力：
- **Memory**：解决“当前上下文之外的信息怎么保留和再次使用”。
- **Planning**：解决“任务太复杂，不能只靠下一步临时决定”的问题。
- **Reflection**：解决“第一次行动或答案可能不够好，需要检查和修正”的问题。
- **Error Handling**：解决“工具调用、参数、外部服务等可能失败”的问题。
- **State / Loop**：保存 Agent 当前处于什么状态，并控制循环什么时候继续、什么时候结束。
## 🔗 一个重要区别
**Tool Calling / Tool Runtime** 更接近“行动接口”；
**Memory / Planning / Reflection** 更接近“让行动变得更持续、更有组织的能力”；
**State / Loop / Error Handling** 则负责让整个系统能够稳定运行。
因此，学习 Agent 组件时不要问“一个 Agent 必须有哪些组件”，而要问：**它为了完成某种任务，遇到了什么限制，于是增加了什么能力？**
## 计划收录
- Tool Calling
- Tool Runtime
- Memory
- Planning
- Reflection
- Error Handling
- State / Loop
重点不是背组件名称，而是理解它们如何共同形成一个可以持续行动的系统。
## 🧪 我实际跑通的理解（2026-09-04）
### Tool Runtime：LLM 决定调用，Runtime 负责执行
我已经用真实的 Qwen API 跑通了一个最小 Tool Calling Agent。
核心关系：
`LLM → Tool Call → Tool Runtime → Tool → Tool Result → LLM`
**LLM 不直接运行 Python 函数。** LLM 返回的是结构化的“我想调用哪个工具 + 参数”，Runtime 再根据工具名找到真正的 Python 函数并执行。
例如 LLM 给出：`search` + `{"keyword": "Python"}`，Runtime 最终执行类似 `search(keyword="Python")`。
### Tool Calling 与 Tool Runtime 的分工
- **Tool Calling**：模型决定“要采取什么工具行动”，并把工具名、参数组织成结构化调用。
- **Tool Runtime**：程序真正执行工具，把结果拿回来，再交给 LLM。
- **Tool**：真正完成具体工作的函数/外部能力。
因此，之前学的 `tool(**arguments)` 很重要：如果 `arguments = {"keyword": "Python"}`，`**arguments` 会把字典解包成关键字参数，相当于 `tool(keyword="Python")`。
### Error Handling：工具可能失败，所以 Runtime 要接住
工具执行不是一定成功的。可能出现参数错误、网络错误、外部服务失败等。
例如：
`try → 执行 Tool → except 捕获异常 → 根据错误类型决定重试 / 返回失败`
这里的 **Exception（异常）** 是程序运行过程中出现的错误状态；`raise ValueError(...)` 可以主动抛出一个异常，`except ValueError as e` 可以捕获它。
对于临时性的网络错误，可以重试；对于明显的参数错误，则通常不应该盲目重试，而应该修正参数或直接把错误反馈给 Agent。
### Agent Loop：不是“调用一次工具”就结束
Agent 的核心是循环：
`Context → LLM → 决策 → Tool Calling → Runtime → Observation/Tool Result → Context → …`
LLM 每次看到当前上下文后，可以决定：继续调用工具，还是直接给最终答案。
我实际跑出的 Demo 中，LLM **一次响应同时返回了两个 Tool Call**：`web_search("Python")` 和 `get_weather("东京")`。Runtime 依次执行两个工具，然后再请求一次 LLM，最终得到回答。
因此要区分两个概念：
- **多个 ****`choices`**：一次 API 请求可以有多个候选输出；我们当前请求通常只产生一个，所以代码取 `response.choices[0]`。
- **多个 ****`tool_calls`**：同一个模型响应里可以要求执行多个工具，因此代码需要 `for tool_call in message.tool_calls`。
### 一个重要认识：Tool Call 多 ≠ LLM 思考轮数多
这次实际运行过程不是：`LLM → Tool A → LLM → Tool B → LLM`。
而是：
`LLM #1 → 同时决定 Tool A + Tool B → Runtime 执行 A/B → LLM #2 → Final Answer`
只有当第二个工具是否调用依赖第一个工具的结果时，才更明显地形成：
`LLM #1 → Tool A → Result A → LLM #2 → Tool B → Result B → LLM #3 → Final`
这也是 ReAct 中“边行动边根据新观察调整”的关键。
### 真实 LLM 接入后，之前的 fake_llm 没有白学
之前的 `fake_llm(messages)` 只是把模型决策模拟出来；换成真实 API 后，整体 Agent Loop 仍然成立：
`真实 LLM → Tool Call → Runtime → Tool Result → 真实 LLM`
所以 Agent 的核心并不是某个 SDK，而是**LLM 的决策能力 + Tool Calling + Runtime + Loop**如何组合。
### 代码里遇到的陌生 Python / SDK 语法
- `**arguments`：关键字参数解包，把字典变成 `func(key=value)` 的形式。
- `Exception`：运行时异常；`try/except` 用于捕获和处理异常。
- **Environment Variable（环境变量）**：程序外部提供的配置，例如 API Key；`.env` 可以保存它，再通过 `load_dotenv()` 和 `os.getenv()` 读取。
- `response.choices[0].message`：不是“Python 魔法”，而是在沿着 API 返回对象的数据结构取值：`response → choices → 第一个 choice → message`。
- `client.chat.completions.create(...)`：调用 SDK 提供的 API 方法，请求模型生成一次响应；参数中 `model` 指定模型，`messages` 提供当前上下文，`tools` 提供模型可使用的工具定义。
### Tool Schema 的位置
LLM 看到的不是 Python 函数本身，而是 **Tool Schema / Tool Definition**：工具名称、描述、参数结构等。它告诉模型“有哪些工具、什么时候可以用、需要什么参数”。
因此可以把关系理解成：
`Python Tool → Tool Schema → LLM → Tool Call → Runtime → Python Tool`
**下一步重点：理解 Tool Schema / Function Calling 为什么能让 LLM 正确表达工具调用，以及真实 API 返回的 ****`tool_call`**** 结构。**
## 🧠 这次学习补充：Context / State / Memory 与 Tool Result
### Context Window：LLM 当前“桌面上的资料”
可以把 Context 理解成：**这一次调用 LLM 时，程序摆给模型看的全部信息**。它可能包括 System Prompt、用户消息、历史消息、Tool Call、Tool Result，以及其他相关资料。
Context 不是永久记忆。LLM 每次调用时只能根据当前提供的 Context 进行生成。
### Token 与 Context Window
- **Token**：模型处理文本时使用的基本计量单位；不能简单等同于“一个汉字”或“一个单词”。
- **Context Window**：一次 LLM 调用能够处理的上下文容量，通常按 token 计。
Agent 特别容易让 Context 变大，因为它可能不断经历：`LLM → Tool → Result → LLM → Tool → Result → …`。如果把所有结果都保留，`messages` 会越来越长。
因此实际系统需要做 **Context Management**，例如截断、总结历史、保留重要信息等；其中 **Compaction / Summarization** 可以理解成“上下文太长了 → 压缩旧内容 → 腾出空间”。
### Tool Result 为什么不能无脑塞进 Context
Tool Result 可能很大，例如搜索工具可能拿到大量网页内容。通常不会把所有原始内容完整塞给 LLM，而会进行筛选、截断、摘要等处理，只保留当前任务真正需要的信息。
这和 RAG 的思想有相似之处：不是把所有知识都交给 LLM，而是尽量提供**少而相关的信息**。
### State / Context / Memory 的区别
- **State**：程序保存的 Agent 当前工作状态，例如 `destination`、`days`、`selected_hotel`、任务是否完成等。它首先是给程序使用的结构化数据。
- **Context / Messages**：当前要提供给 LLM 的信息，也就是模型“现在能看到什么”。
- **Memory**：需要在当前任务/当前上下文之外长期保存，并在以后再次使用的信息。
它们不是固定的数据结构，而是不同的职责。实际框架可能把 `messages` 放进 State，也可能单独管理。
### 一个最小的数据流
`用户 → State / Messages → LLM → Tool Call → Runtime → Tool → Tool Result → Messages / State → LLM → Final Answer`
这里最重要的是：**LLM 负责决定下一步，Runtime 负责执行，Tool 负责真正干活；Tool Result 回来后，程序把它放入下一轮 LLM 能看到的 Context。**
### Agent 的“自主性”从哪里来？
工具本身没有思想。程序也不需要理解 LLM 为什么这么想。LLM 根据当前 Context 和可用 Tool Schema 决定“下一步做什么”，Runtime 再把这个决定执行出来。
可以理解为：
- **LLM = 决策者**：决定是否调用工具、调用哪个工具、参数是什么。
- **Runtime = 执行协调者**：找到工具、校验/处理调用、执行并返回结果。
- **Tool = 外部能力**：真正完成搜索、计算、查询等工作。
因此 Agent 更像是“**软决策 + 硬约束**”：LLM 提出行动建议，程序通过 Schema、参数校验、状态判断等机制限制和约束实际执行。
### Tool Schema：LLM 与 Python Tool 之间的桥
Tool Schema / Tool Definition 不是 Tool 本身，而是给 LLM 看的“工具说明书”：工具名、用途、参数名称、参数类型、必填参数等。
例如 `search_web` 可以声明 `query` 是 `string`。LLM 根据这个 Schema 生成结构化 Tool Call；Runtime 再把这个调用映射到真正的 Python 函数。
关系可以记成：
`Python Tool → Tool Schema → LLM → Tool Call → Runtime → Python Tool`
Schema 是指路，不是魔法。LLM 仍可能产生错误参数，因此可靠系统还需要 Runtime / Schema 校验、错误处理和重试策略。
### Python：`**arguments`
如果 `arguments = {"city": "东京"}`，那么 `tool(**arguments)` 相当于 `tool(city="东京")`。`**` 在这里是把字典解包成关键字参数。
类似地，`*args` 是把列表/元组等解包成位置参数。
### `llm(...)` 为什么在 Loop 里面？
Agent Loop 中的 `messages` 会随着 Tool Result 不断变化，所以不能只在循环外调用一次 LLM。
`LLM → Tool → Result → messages 更新 → 再次 LLM` 才能让模型看到刚刚获得的新信息，并决定下一步。`tools` 定义可以不变，但 `messages` 会变化。
## 🧭 当前学习位置
已经能够从概念上读懂最小 Agent Loop，并理解：LLM API、Tool Schema、Tool Calling、Tool Runtime、Tool Result、State、Context、Context Window、Token，以及它们之间的数据流。
下一阶段可以进入真实 Agent 的工程问题：**Tool Schema / Function Calling 的真实 API 返回结构、Error Handling、Retry、循环上限，以及随后做一个最小可运行 Agent。**