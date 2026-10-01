
from os import getenv

from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any

from config import Config

# 加载 .env 文件中的环境变量
load_dotenv()

class LLM:
    def __init__(self, model: str=None, api_key: str=None, base_url: str=None, config: Config=None):
        if config is None:
            config = Config.from_env()
        self.config = config
        self.model = model or config.default_model
        api_key = api_key or config.api_key
        base_url = base_url or config.base_url

        if not all([self.model, api_key, base_url]):
            missing = [name for name, val in [
                ("LLM_MODEL_ID", self.model),
                ("LLM_API_KEY", api_key),
                ("LLM_BASE_URL", base_url),
            ] if not val]
            raise ValueError(f"缺少配置: {', '.join(missing)}，请在 .env 文件中配置")

        self.client = OpenAI(api_key=api_key,base_url=base_url)

    def chat(self, messages: List[Dict[str, str]], **kwargs):
        """调用模型，返回原始的 message 对象（含 content 和 tool_calls）。
        需要处理工具调用时必须用这个方法，因为 tool_calls 不在 content 里"""
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, **kwargs
        )
        return response.choices[0].message

    def think(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """只取文字内容（适合不需要工具调用的场景）"""
        return self.chat(messages, **kwargs).content

    def stream(self, messages: List[Dict[str, str]], **kwargs):
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, stream=True, **kwargs
        )

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


if __name__=='__main__':
    llm = LLM()
    for piece in llm.stream([
        {"role": "user", "content": "用一句话介绍你自己"}
    ]):
        print(piece, end="", flush=True)
