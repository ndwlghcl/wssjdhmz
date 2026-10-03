import time

import openai

# openai API 层的错误分类 + 通用重试引擎。
# 工具执行和 LLM 调用都要对临时性错误做退避重试，逻辑相同，抽成公共引擎；
# 唯一差异是"重试用尽之后怎么办"，由各层自己的契约决定：
# - 工具层（execute_tool）：永不抛异常，错误转成观察结果喂回模型
# - LLM 层（llm.chat/stream）：模型自己挂了，没有"喂回错误"的通道，只能抛给调用方兜底
# with_retry 的默认行为定为中性的"抛出"，用尽策略留给调用方表达。

# 临时性错误（重试有意义）：超时、网络连接、限流、服务端 5xx
# 说明：APITimeoutError 本身是 APIConnectionError 的子类，这里显式列出只为可读性
TRANSIENT_EXCEPTIONS = (
    openai.APITimeoutError,        # 请求超时
    openai.APIConnectionError,     # 网络连接问题
    openai.RateLimitError,         # 429 限流
    openai.InternalServerError,    # 5xx 服务端临时故障
)

MAX_RETRIES = 3        # 临时性错误最多额外重试次数（不含第一次）
RETRY_BASE_DELAY = 1.0 # 首次重试前等待秒数，之后指数退避：1s → 2s → 4s


def with_retry(fn, *, what: str = "调用", max_retries: int = MAX_RETRIES, base_delay: float = RETRY_BASE_DELAY):
    """执行 fn()，临时性错误自动指数退避重试；重试用尽或永久性错误都向上抛出。
    永久性错误（400 参数非法、401 认证失败等）第一次就直接放行，不浪费重试。"""
    attempt = 0
    while True:
        try:
            return fn()
        except TRANSIENT_EXCEPTIONS as e:
            if attempt >= max_retries:
                raise
            attempt += 1
            delay = base_delay * (2 ** (attempt - 1))   # 1s, 2s, 4s…
            print(f"⚠️ {what} 临时失败（{type(e).__name__}），{delay:.0f}s 后进行第 {attempt} 次重试…")
            time.sleep(delay)
