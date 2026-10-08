
from config import Config
from llm import LLM


# def rerank(query,documents,top_k=5):
#     config=Config.from_env()
#

import os, requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("LLM_API_KEY")

documents = []

resp = requests.post(
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    json={
        "model": "qwen3.7-text-rerank",
        "input": {
            "query": "Chunking 有哪些策略？",
            "documents": documents,
        },
        "parameters": {"top_n": 3, "return_documents": True},
    },
)
print(resp.status_code)
print(resp.json())
