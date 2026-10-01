from agent import Agent
from config import Config
from get_current_time import get_current_time
from llm import LLM
from message import Message
from tool_executor import ToolExecutor


class SimpleAgent(Agent):

    def __init__(self, name: str, system_prompt: str | None = None, config: Config | None = None):
        if config is None:
            config = Config.from_env()
        # 把同一个 config 传进去，避免 LLM 内部再建一个配置实例
        llm = LLM(config=config)

        super().__init__(name, llm, system_prompt, config)

    # def _stream_reply(self, messages: list[Message]) -> str:
    #     """流式输出回复并收集完整内容，同时存入历史"""
    #     pieces: list[str] = []
    #     for piece in self.stream_with_config(messages, self.config):
    #         print(piece, end="", flush=True)
    #         pieces.append(piece)
    #     print()
    #
    #     reply = "".join(pieces)
    #     self.add_message(Message(role="assistant", content=reply))
    #     return reply

    def run(self, input_text: str) -> str:
        # 1. 用户输入存入历史
        self.add_message(Message(role="user", content=input_text))

        # 2. 组装请求消息：system_prompt 只在发请求时注入，不存入历史
        messages = self.get_history()
        if self.system_prompt:
            system_msg = Message(role="system", content=self.system_prompt)
            messages = [system_msg] + messages

            # print(messages)

        # 3. 流式输出并收集完整回复
        return self._stream_reply(messages)

    def greet(self) -> str:
        messages = [Message(role="system", content="说一句欢迎语。")]
        return self._stream_reply(messages)



if __name__ == '__main__':
    agent = SimpleAgent(name="小助手", system_prompt="你是一个简洁的中文助手。")
    agent.greet()
    print("输入 exit 退出")
    while True:
        user_input = input("\n你: ")
        if user_input.strip().lower() in ("exit", "quit", "退出"):
            break
        print("助手: ", end="")
        agent.run(user_input)
