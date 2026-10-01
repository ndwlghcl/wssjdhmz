import json

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


class ReActAgent(Agent):

    def __init__(self, name: str, config: Config | None = None):
        if config is None:
            config = Config.from_env()
        llm = LLM(config=config)

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

    def _clip_observation(self, text: str) -> str:
        """限制单条工具结果的长度：超长的检索原文会让历史 token 快速膨胀。
        这里用字符数近似 token 数，中文场景够用"""
        limit = self.config.max_obs_tokens
        if len(text) > limit:
            return text[:limit] + f"…（已截断，原长 {len(text)} 字）"
        return text

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
                # execute_tool 内部已做"临时错误重试 + 永久错误兜底"，且保证不抛异常，
                # 所以这里直接拿返回值即可，返回值一定是字符串（正常结果或错误说明）
                result = self.tool_executor.execute_tool(name, **args)
                # 存入历史前对超长观察结果截断，避免多轮对话后历史 token 无限膨胀
                obs = self._clip_observation(str(result))
                print(f"👁 观察结果: {obs}")

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