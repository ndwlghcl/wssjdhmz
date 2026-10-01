from get_current_time import get_current_time
from tool import Tool


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
        """按名字执行工具。任何错误都不抛出，而是返回错误文字——
        因为在 Agent 循环里，工具报错也要变成'观察结果'喂给模型，不能让整个程序崩掉"""
        tool = self.tools.get(name)
        if tool is None:
            return f"错误: 不存在名为 {name} 的工具"
        try:
            return tool.run(**kwargs)
        except Exception as e:
            return f"工具执行出错: {e}"

if __name__ == "__main__":
    executor = ToolExecutor()
    # 装饰后的 get_current_time 已经是 FunctionTool 对象，直接注册
    executor.register_tool(get_current_time)

    print(executor.get_available_tools_str())
    print(executor.execute_tool("get_current_time"))
    # 顺便测一下错误路径：工具不存在时不崩、返回错误文字
    print(executor.execute_tool("不存在的工具"))
