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
"""


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

    def run(self, input_text: str) -> str:
        # 对话消息列表：这是真正的协议载体，不再需要拼提示词模板
        messages: list = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input_text},
        ]
        tools = self.tool_executor.get_tools_schema()

        for step in range(1, self.max_steps + 1):
            print(f"\n--- 第 {step} 步 ---")

            message = self.llm.chat(messages, tools=tools, temperature=0)

            # 没有工具调用 → 模型的 content 就是最终答案
            if not message.tool_calls:
                answer = message.content or ""
                print(answer)
                return answer

            # 有工具调用：先把助手这条消息原样存回历史
            # （必须存，因为下面 role="tool" 的消息要引用它的 id，否则 API 会报错）
            messages.append(message)

            for call in message.tool_calls:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments)
                except json.JSONDecodeError as e:
                    # 参数不是合法 JSON：把错误当成工具结果喂回去，让模型自己纠正
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": f"参数解析失败: {e}",
                    })
                    continue

                print(f"🔧 调用工具: {name}，参数: {args}")
                result = self.tool_executor.execute_tool(name, **args)
                print(f"👁 观察结果: {result}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": str(result),
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
        answer = agent.run(question)
        print(f"\n✅ 最终答案: {answer}")
