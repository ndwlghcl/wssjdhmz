import requests
from openai import OpenAI
from typing import List, Dict, Any

from config import Config
from retry import with_retry


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
        需要处理工具调用时必须用这个方法，因为 tool_calls 不在 content 里。
        临时性错误自动退避重试；重试用尽向上抛出——模型挂了没有"把错误喂回模型"的
        通道，只能让调用方（Agent 循环）兜底，这点和工具层"永不抛异常"正好相反"""
        response = with_retry(
            lambda: self.client.chat.completions.create(
                model=self.model, messages=messages, **kwargs
            ),
            what="chat",
        )
        return response.choices[0].message

    def think(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """只取文字内容（适合不需要工具调用的场景）"""
        return self.chat(messages, **kwargs).content

    def stream(self, messages: List[Dict[str, str]], **kwargs):
        response = with_retry(
            lambda: self.client.chat.completions.create(
                model=self.model, messages=messages, stream=True, **kwargs
            ),
            what="stream",
        )

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def embed(self, input: str | list[str]):
        """把文字转成向量，input 可以是单条 str 或批量 list[str]，返回原始响应。
        embedding 同样是网络调用，会超时/限流，所以和 chat 走同一个重试引擎。
        （之前 RAG_tool 直接裸用 llm.client.embeddings，重试罩不到这里）"""
        return with_retry(
            lambda: self.client.embeddings.create(
                model=self.config.embedding_model, input=input
            ),
            what="embedding",
        )

    def rerank(self, query: str, documents: list[str], top_n: int = 5) -> list[dict]:
        """调用排序模型，返回 [{"index": i, "relevance_score": s}, ...]（按分数降序）。
        rerank 是 DashScope 私有协议（OpenAI 协议里没有这个端点），所以用 requests 直发，
        不走 self.client；用 json= 时 Content-Type 会自动设置，不用手写"""
        def call():
            resp = requests.post(
                self.config.rerank_model_url,
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                json={
                    "model": self.config.rerank_model,
                    "input": {"query": query, "documents": documents},
                    "parameters": {"top_n": top_n},
                },
                timeout=self.config.request_timeout,
            )
            resp.raise_for_status()      # 关键：requests 默认对 4xx/5xx 不报错，不主动抛就永远静默失败
            return resp.json()["output"]["results"]   # 结果在 output.results 里，不是顶层
        return with_retry(call, what="rerank")


if __name__=='__main__':
    llm = LLM()
    for piece in llm.stream([
        {"role": "user", "content": "用一句话介绍你自己"}
    ]):
        print(piece, end="", flush=True)
