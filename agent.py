from abc import ABC, abstractmethod
from typing import Any

from config import Config
from llm import LLM


class Agent(ABC):
    """Agent 基类：公共部分是"配置 + 记忆 + 流式输出"，具体怎么跑由 run() 定义"""

    def __init__(self, name: str, llm: LLM, system_prompt: str | None = None, config: Config | None = None):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config.from_env()
        # 历史统一用协议原生的字典表示（不再包装 Message 类）
        self.messages: list[dict] = []
        if system_prompt:
            self.add_message({"role": "system", "content": system_prompt})

    @abstractmethod
    def run(self, input_text: str) -> str:
        pass

    def add_message(self, message: dict):
        self.messages.append(message)
        # 记忆管理：超过上限时裁掉最旧的消息
        # if len(self._history) > self.config.max_history_length:
        #     self._history = self._history[-self.config.max_history_length:]

    def clear_history(self):
        self.messages = []
        if self.system_prompt:
            self.add_message({
                "role": "system",
                "content": self.system_prompt,
            })


    def get_history(self) -> list[dict]:
        return self.messages.copy()

    def stream_with_config(self, messages: list[dict], config: Config | None = None):
        if config is None:
            config = self.config
        params: dict[str, Any] = {"temperature": config.temperature}
        if config.max_tokens is not None:
            params["max_tokens"] = config.max_tokens
        return self.llm.stream(messages=messages, **params)

    def _stream_reply(self, messages: list[dict]) -> str:
        """流式输出回复并收集完整内容，同时存入历史"""
        pieces: list[str] = []
        for piece in self.stream_with_config(messages, self.config):
            print(piece, end="", flush=True)
            pieces.append(piece)
        print()

        reply = "".join(pieces)
        self.add_message({"role": "assistant", "content": reply})
        return reply

    def __str__(self) -> str:
        return f"Agent(name={self.name}, provider={self.config.default_provider})"
