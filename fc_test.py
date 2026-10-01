"""最小验证：OpenAI 原生 function calling 返回的数据长什么样"""
import json

from llm import LLM
from RAG_tool import retrieval
from tool_executor import ToolExecutor

llm = LLM()
executor = ToolExecutor()
executor.register_tool(retrieval)

print("=== 工具 schema ===")
print(json.dumps(executor.get_tools_schema(), ensure_ascii=False, indent=2))

messages = [{"role": "user", "content": "俄耳甫斯为什么回头？"}]

resp = llm.chat(messages, tools=executor.get_tools_schema())
message = resp

print("\n=== 模型返回的 content ===")
print(repr(message.content))

print("\n=== tool_calls ===")
print(message.tool_calls)
print(message.model_dump())

if message.tool_calls:
    call = message.tool_calls[0]
    print("\n=== 第一个调用 ===")
    print("id:", call.id)
    print("name:", call.function.name)
    print("arguments（原始字符串）:", call.function.arguments)
    print("arguments（解析后）:", json.loads(call.function.arguments))

print(message.model_dump())

