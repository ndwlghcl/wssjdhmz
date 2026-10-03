import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable, get_args, get_origin

# Python 类型 → JSON Schema 类型的对照表
_JSON_TYPE = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _annotation_to_schema(annotation) -> dict:
    """把单个类型标注翻译成 JSON Schema 片段，有两种标注要先剥开再看：
    1. Annotated[真实类型, "描述"]：描述写进 schema 的 description 给模型看，
       真实类型继续往下翻译（__metadata__ 只在 Annotated 上存在，靠它识别）
    2. 泛型下标（list[str] 等）：标注对象不等于 list 本身，直接查表会 miss，
       要先用 get_origin 剥出骨架再查，get_args 拿到元素类型填进 items"""
    description = None
    if getattr(annotation, "__metadata__", None):
        description = next((m for m in annotation.__metadata__ if isinstance(m, str)), None)
        annotation = get_args(annotation)[0]

    origin = get_origin(annotation)
    if origin is None:
        schema = {"type": _JSON_TYPE.get(annotation, "string")}
    else:
        schema = {"type": _JSON_TYPE.get(origin, "string")}
        if origin is list:
            args = get_args(annotation)
            if args:
                schema["items"] = _annotation_to_schema(args[0])

    if description is not None:
        schema["description"] = description
    return schema


def build_parameters_schema(func: Callable) -> dict:
    """根据函数的参数和类型标注，自动生成 JSON Schema。
    这样用 @tool 时不需要手写参数定义，框架照着函数签名生成即可"""
    properties = {}
    required = []

    for name, param in inspect.signature(func).parameters.items():
        # 跳过 *args / **kwargs 这类参数
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        properties[name] = _annotation_to_schema(param.annotation)
        # 没有默认值的参数 = 必填
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {"type": "object", "properties": properties, "required": required}


class Tool(ABC):
    """工具基类：执行器只认识这个接口，但用户通常不需要直接继承它"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.parameters: dict = {"type": "object", "properties": {}}

    @abstractmethod
    def run(self, **kwargs) -> Any:
        """执行工具，返回结果"""

    def to_dict(self) -> dict:
        """转成 OpenAI function calling 需要的 JSON Schema"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class FunctionTool(Tool):
    """把一个普通函数包装成 Tool。
    name 缺省用函数名，description 缺省用 docstring，参数 schema 从函数签名自动推导"""

    def __init__(self, func: Callable, name: str | None = None, description: str | None = None):
        super().__init__(
            name=name or func.__name__,
            description=description or (func.__doc__ or "").strip() or "（无描述）",
        )
        self.func = func
        # 从函数签名推导参数结构，模型靠它知道该传什么参数
        self.parameters = build_parameters_schema(func)

    def run(self, **kwargs) -> Any:
        return self.func(**kwargs)


def tool(name: str | None = None, description: str | None = None) -> "Callable[[Callable], FunctionTool]":
    """装饰器：把普通函数变成工具，省去为每个工具写一个类的麻烦

    用法：
        @tool(description="获取当前时间")
        def get_current_time() -> str:
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    参数含义建议用 Annotated 补充，会自动写进 schema 的 description，模型填参更准：
        @tool(description="检索知识库")
        def retrieval(input: Annotated[str, "要检索的问题，完整问句"]) -> str: ...
    """
    def decorator(func: Callable) -> FunctionTool:
        return FunctionTool(func, name=name, description=description)
    return decorator
