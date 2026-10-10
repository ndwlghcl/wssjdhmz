from retry import ALL_TRANSIENT_EXCEPTIONS, with_retry

from get_current_time import get_current_time
from tool import Tool

# schema 的 JSON 类型 → 用来 isinstance 检查的 Python 类型
# 注意 number 放行 int（JSON 里 1 也是合法的 number）
_JSON_PY_TYPES = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


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

    def validate_args(self, name: str, args) -> str | None:
        """bind 校验：把模型给出的参数绑定到工具 schema 上检查，返回错误说明，None = 通过。
        json.loads 只保证"是合法 JSON"，挡不住这些错：不是字典、缺必填参数、
        编造参数名、类型给错。放过去的话会在 tool.run(**kwargs) 时炸成 Python 的
        TypeError，模型收到的是内部报错文本，很难自我纠正；这里提前对照 schema
        拦下，错误字符串由 Agent 循环当作 tool 消息喂回模型，让它自己改。
        与 execute_tool 的分工：本方法管"参数对不对"，execute_tool 管"执行中出什么错"。"""
        tool = self.tools.get(name)
        if tool is None:
            return f"错误: 不存在名为 {name} 的工具"

        # json.loads 成功 ≠ 是字典：模型可能输出 "foo"、[1,2] 这类合法 JSON
        if not isinstance(args, dict):
            return f"参数校验失败: 参数必须是 JSON 对象，收到了 {type(args).__name__}"

        props = tool.parameters.get("properties", {})
        required = tool.parameters.get("required", [])

        missing = [k for k in required if k not in args]
        unknown = [k for k in args if k not in props]
        if missing or unknown:
            parts = []
            if missing:
                parts.append(f"缺少必填参数: {', '.join(missing)}")
            if unknown:
                parts.append(f"未知参数: {', '.join(unknown)}（可用参数: {', '.join(props)}）")
            return "参数校验失败: " + "；".join(parts)

        for key, value in args.items():
            expected = props[key].get("type")
            if expected is None:
                continue
            # bool 是 int 的子类：isinstance(True, int) 为 True，
            # 所以除 boolean 外的任何类型都要先排除 bool，否则 True 会冒充 1
            if isinstance(value, bool) and expected != "boolean":
                return f"参数校验失败: 参数 {key} 应为 {expected} 类型，收到了 boolean"
            py_type = _JSON_PY_TYPES.get(expected)
            if py_type is not None and not isinstance(value, py_type):
                return f"参数校验失败: 参数 {key} 应为 {expected} 类型，收到了 {type(value).__name__}"

        return None

    def execute_tool(self, name: str, **kwargs):
        """按名字执行工具。错误分两类处理：
        - 临时性错误（网络/超时/限流/5xx）：with_retry 自动指数退避重试，大概率能自愈
        - 永久性错误（400 参数非法、401 认证失败等）或重试用尽：返回错误文字，不抛出
        无论哪种情况都不抛异常——因为在 Agent 循环里，工具报错要变成'观察结果'喂给模型，
        不能让整个程序崩掉。
        （与 llm.chat 的分工：同一个重试引擎，但 LLM 挂了没有喂回通道，
        那边用尽后是把异常抛给调用方兜底）"""
        tool = self.tools.get(name)
        if tool is None:
            return f"错误: 不存在名为 {name} 的工具"

        try:
            # 这里用默认的 openai 侧重试分类：工具内部若自己用了别的客户端（retrieval → rerank
            # 走 requests），它在自己那层已经重试过了；这里再重试，尝试次数会变成相乘
            return with_retry(lambda: tool.run(**kwargs), what=f"工具 {name}")
        except ALL_TRANSIENT_EXCEPTIONS as e:
            # 临时错误能走到这里 = 某一层的 with_retry 已经退避重试用尽
            # （这里用全集判断是为了"报告"准确，不是为了重试——见 retry.py 里该常量的注释）
            return f"工具执行出错（已重试多次仍未成功）: {type(e).__name__}: {e}"
        except Exception as e:
            # 永久性错误：with_retry 第一次就放行，没重试过，重试无意义，直接返回错误文字
            return f"工具执行出错: {type(e).__name__}: {e}"


if __name__ == "__main__":
    executor = ToolExecutor()
    # 装饰后的 get_current_time 已经是 FunctionTool 对象，直接注册
    executor.register_tool(get_current_time)

    print(executor.get_available_tools_str())
    print(executor.execute_tool("get_current_time"))
    # 顺便测一下错误路径：工具不存在时不崩、返回错误文字
    print(executor.execute_tool("不存在的工具"))
    # bind 校验：无参工具收到任何参数都该被拦下，工具不存在也有明确返回
    print(executor.validate_args("get_current_time", {"x": 1}))
    print(executor.validate_args("不存在的工具", {}))