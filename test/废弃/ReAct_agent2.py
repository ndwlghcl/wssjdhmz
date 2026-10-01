import json
import time

from agent import Agent
from config import Config
from get_current_time import get_current_time
from llm import LLM
from RAG_tool import retrieval
from tool_executor import ToolExecutor

SYSTEM_PROMPT = """你是一个可以使用工具的智能助手。

规则：
1. 回答任何知识性问题之前，必须先调用 retrieval 工具检索知识库，禁止仅凭自己已有的知识作答
2. 检索结果带有相似度分数，优先使用分数高的资料
3. 如果检索结果为空或明显与问题无关，说明资料不足，不要编造答案
4. 同一个问题最多检索两次，不要反复换问法重试
5. 问题与知识库无关时（例如询问时间），可以调用相应的工具或直接回答
6. 使用检索资料作答时，要在相关结论后标注来源，格式为（来源：文件名 > 章节名）。检索结果里已给出"来源"和"章节"字段，直接引用即可，不要编造来源
"""

# 单条工具观察结果存入历史时的最大字符数：防止超长检索原文让对话历史 token 膨胀。
MAX_OBS_CHARS = 2000

# ===== 工具调用重试配置 =====
TOOL_MAX_RETRIES = 3          # 遇到临时性错误时，最多额外重试几次（不含第一次）
TOOL_RETRY_BASE_DELAY = 1.0   # 首次重试前等待秒数，之后按指数退避：1s → 2s → 4s

# 视为"临时性、值得重试"的错误特征（网络波动、超时、限流、服务端临时故障）
TRANSIENT_ERROR_HINTS = (
    "timeout", "timed out", "connection", "network", "temporarily",
    "rate limit", "429", "500", "502", "503", "504",
    "reset", "unavailable", "refused", "eof occurred",
)


class ReActAgent(Agent):

    def __init__(self, name: str, config: Config | None = None):
        if config is None:
            config = Config.from_env()
        llm = LLM()

        super().__init__(name, llm, system_prompt=SYSTEM_PROMPT, config=config)
        self.tool_executor = ToolExecutor()
        self.tool_executor.register_tool(get_current_time)
        self.tool_executor.register_tool(retrieval)
        self.max_steps = 5

        # 对话历史：全部用字典存（便于裁剪、序列化，也不会夹带 reasoning_content 之类的私有字段）
        self.messages: list[dict] = [{"role": "system", "content": self.system_prompt}]

    def _trim_history(self):
        """裁剪历史。两条策略：
        1. system 永远保留，不参与裁剪
        2. 裁剪后第一条业务消息必须是 user —— 一轮完整对话总是以 user 开始，
           所以切在 user 后面，绝不会切断 assistant(tool_calls) 和 tool 的配对
        """
        limit = self.config.max_history_length
        if len(self.messages) <= limit:
            return

        system = self.messages[0]
        # 注意：limit 为 1 时 (limit - 1) 是 0，而 [-0:] 会取整个列表，
        # 所以这里用 max(..., 1) 兜底
        keep = max(limit - 1, 1)
        recent = self.messages[-keep:]

        # 从前往后丢，直到第一条是 user
        while recent and recent[0].get("role") != "user":
            recent = recent[1:]

        self.messages = [system] + recent
        print(f"（历史已裁剪至 {len(self.messages)} 条）")

    def _assistant_payload(self, message) -> dict:
        """把 SDK 返回的助手消息转成最小字典：只保留协议要求的字段"""
        payload: dict = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in message.tool_calls
            ]
        return payload

    def _is_transient(self, exc: Exception) -> bool:
        """粗略判断异常是否属于"临时性、重试可能成功"的类型。
        网络波动、超时、限流、服务端 5xx 都值得重试；
        参数非法、逻辑错误这类"永久性错误"重试也没用，应直接交给模型。"""
        text = f"{type(exc).__name__}: {exc}".lower()
        return any(hint in text for hint in TRANSIENT_ERROR_HINTS)

    def _execute_with_retry(self, name: str, args: dict):
        """执行工具，遇到临时性错误自动重试（指数退避）；
        只有"永久性错误"或"重试用尽仍失败"时，才把错误作为观察结果返回给模型。"""
        attempt = 0
        while True:
            try:
                return self.tool_executor.execute_tool(name, **args)
            except Exception as e:
                transient = self._is_transient(e)
                # 永久性错误，或重试次数已用尽 → 交给模型处理
                if not transient or attempt >= TOOL_MAX_RETRIES:
                    if transient:
                        return f"工具执行失败（已重试 {attempt} 次仍未成功）: {type(e).__name__}: {e}"
                    return f"工具执行失败: {type(e).__name__}: {e}"
                # 临时性错误且还有重试机会 → 等待后重试
                attempt += 1
                delay = TOOL_RETRY_BASE_DELAY * (2 ** (attempt - 1))   # 1s, 2s, 4s…
                print(f"⚠️ 工具 {name} 临时失败（{type(e).__name__}），{delay:.0f}s 后进行第 {attempt} 次重试…")
                time.sleep(delay)

    def run(self, input_text: str) -> str:
        self._trim_history()
        self.messages.append({"role": "user", "content": input_text})

        tools = self.tool_executor.get_tools_schema()

        for step in range(1, self.max_steps + 1):
            print(f"\n--- 第 {step} 步 ---")

            message = self.llm.chat(self.messages, tools=tools, temperature=0)

            # 没有工具调用 → 模型的 content 就是最终答案
            if not message.tool_calls:
                answer = message.content or ""
                print(answer)
                self.messages.append({"role": "assistant", "content": answer})
                return answer

            # 有工具调用：先把助手消息存回历史
            # （必须存，因为下面 role="tool" 的消息要引用它的 id）
            self.messages.append(self._assistant_payload(message))

            for call in message.tool_calls:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments)
                except json.JSONDecodeError as e:
                    # 参数不是合法 JSON：把错误当成工具结果喂回去，让模型自己纠正
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": f"参数解析失败: {e}",
                    })
                    continue

                print(f"🔧 调用工具: {name}，参数: {args}")
                # B + 重试：临时性错误先自动重试，重试用尽或永久错误才回喂模型
                result = self._execute_with_retry(name, args)
                print(f"👁 观察结果: {result}")

                # E：存入历史前对超长观察结果截断，避免多轮对话后历史 token 无限膨胀
                obs = str(result)
                if len(obs) > MAX_OBS_CHARS:
                    obs = obs[:MAX_OBS_CHARS] + f"…（已截断，原长 {len(obs)} 字）"

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": obs,
                })

        return "抱歉，我尝试了多次仍未能完成任务。"


if __name__ == "__main__":
    agent = ReActAgent(name="ReAct助手")
    print("输入 exit 退出")
    while True:
        question = input("\n你: ")
        if question.strip().lower() in ("exit", "quit", "退出"):
            break
        print("助手: ", end="")
        agent.run(question)