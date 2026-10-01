from datetime import datetime

from tool import tool


# 一个普通函数 + 一个装饰器，就注册成工具了，不需要写类
@tool(name="get_current_time", description="获取当前的日期和时间，无需参数")
def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
