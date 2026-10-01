import time

import openai

from get_current_time import get_current_time
from tool import Tool

# ===== 工具调用重试配置 =====
TOOL_MAX_RETRIES = 3          # 临时性错误最多额外重试次数（不含第一次）
TOOL_RETRY_BASE_DELAY = 1.0   # 首次重试前等待秒数，之后指数退避：1s → 2s → 4s

# 临时性错误（重试有意义）：超时、网络连接、限流、服务端 5xx
# 说明：APITimeoutError 本身是 APIConnectionError 的子类，这里显式列出只为可读性
TRANSIENT_EXCEPTIONS = (
    openai.APITimeoutError,        # 请求超时
    openai.APIConnectionError,     # 网络连接问题
    openai.RateLimitError,         # 429 限流
    openai.InternalServerError,    # 5xx 服务端临时故障
)


class ToolExecutor:
    """工具注册器：负责登记所有工具、按名字执行"""

    def __init__(self):
        # 用字典存，按名字查找比遍历列表快，也不会出现重名工具
        self.tools: dict[str, Tool] = {}

    def register_tool(self, tool: Tool):
        self.tools[tool.name] = tool

    def get_tool(self, name: str) -> Tool | None:
        return self.tools.get(name)

    def get_available_tools_str(self) -> str:
        """（旧的文本协议用法，function calling 用不到，保留是为了兼容 test.py）"""
        return "\n".join(
            f"- {tool.name}: {tool.description}" for tool in self.tools.values()
        )

    def get_tools_schema(self) -> list[dict]:
        """导出所有工具的 JSON Schema，作为 tools 参数传给模型"""
        return [tool.to_dict() for tool in self.tools.values()]

    def execute_tool(self, name: str, **kwargs):
        """按名字执行工具。错误分两类处理：
        - 临时性错误（网络/超时/限流/5xx）：自动重试若干次（指数退避），大概率能自愈
        - 永久性错误（400 参数非法、401 认证失败等）或重试用尽：返回错误文字，不抛出
        无论哪种情况都不抛异常——因为在 Agent 循环里，工具报错要变成'观察结果'喂给模型，
        不能让整个程序崩掉"""
        tool = self.tools.get(name)
        if tool is None:
            return f"错误: 不存在名为 {name} 的工具"

        attempt = 0
        while True:
            try:
                return tool.run(**kwargs)
            except TRANSIENT_EXCEPTIONS as e:
                # 临时性错误：还有重试机会就退避后重试
                if attempt >= TOOL_MAX_RETRIES:
                    return f"工具执行出错（已重试 {attempt} 次仍未成功）: {type(e).__name__}: {e}"
                attempt += 1
                delay = TOOL_RETRY_BASE_DELAY * (2 ** (attempt - 1))   # 1s, 2s, 4s…
                print(f"⚠️ 工具 {name} 临时失败（{type(e).__name__}），{delay:.0f}s 后进行第 {attempt} 次重试…")
                time.sleep(delay)
            except Exception as e:
                # 永久性错误或未知错误：重试无意义，直接返回错误文字交给模型
                return f"工具执行出错: {type(e).__name__}: {e}"


if __name__ == "__main__":
    executor = ToolExecutor()
    # 装饰后的 get_current_time 已经是 FunctionTool 对象，直接注册
    executor.register_tool(get_current_time)

    print(executor.get_available_tools_str())
    print(executor.execute_tool("get_current_time"))
    # 顺便测一下错误路径：工具不存在时不崩、返回错误文字
    print(executor.execute_tool("不存在的工具"))