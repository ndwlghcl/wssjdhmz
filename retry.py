import time

import openai
import requests

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

# requests 侧：不走 openai SDK 的调用用这个（目前只有 llm.rerank——rerank 是 DashScope
# 私有协议，只能用 requests 直发）。
# 教训：临时性错误要按"语义"分类，不能按"哪个库抛的"分类。这里原来只收 openai 的异常类，
# 于是 rerank 外面那层 with_retry 一次都不会触发——白套了一层保护，而且看不出来。
TRANSIENT_REQUESTS_EXCEPTIONS = (
    requests.exceptions.Timeout,          # 连接/读取超时（ConnectTimeout、ReadTimeout 都是它的子类）
    requests.exceptions.ConnectionError,  # DNS 失败、连接被拒、断网
    # HTTPError 本身横跨 4xx/5xx，放进来是安全的：rerank 只在"自己判定为 5xx"时才抛它，
    # 4xx 走另一条分支抛 ValueError（永久错误不该重试）
    requests.exceptions.HTTPError,
)

# "所有已知的临时错误"——只用于**分类/报告**，不要拿去当重试清单。
# 这个区别很重要：execute_tool 包的是任意工具，它需要认得出"这是临时错误"才能给出准确文案；
# 但绝不能因此去重试 requests 错误——工具内部（如 retrieval → llm.rerank）已经各自重试过了，
# 外层再重试一遍，尝试次数就变成相乘（4 次 × 4 次 = 16 次真实请求），故障期间等于帮着打服务端。
ALL_TRANSIENT_EXCEPTIONS = TRANSIENT_EXCEPTIONS + TRANSIENT_REQUESTS_EXCEPTIONS

MAX_RETRIES = 3        # 临时性错误最多额外重试次数（不含第一次）
RETRY_BASE_DELAY = 1.0 # 首次重试前等待秒数，之后指数退避：1s → 2s → 4s


def with_retry(fn, *, what: str = "调用", max_retries: int = MAX_RETRIES,
               base_delay: float = RETRY_BASE_DELAY, exceptions=TRANSIENT_EXCEPTIONS):
    """执行 fn()，临时性错误自动指数退避重试；重试用尽或永久性错误都向上抛出。
    永久性错误（400 参数非法、401 认证失败等）第一次就直接放行，不浪费重试。

    exceptions：哪一类异常算"临时"。默认是 openai 侧的分类，所以 chat / stream / embed /
    工具执行的行为与改动前完全一致；用别的 HTTP 客户端的调用要显式传入自己的分类
    （见 llm.rerank 传 TRANSIENT_REQUESTS_EXCEPTIONS）。
    可以直接给元组，也可以给单个异常类（except 两种都支持）。"""
    attempt = 0
    while True:
        try:
            return fn()
        except exceptions as e:
            if attempt >= max_retries:
                raise
            attempt += 1
            delay = base_delay * (2 ** (attempt - 1))   # 1s, 2s, 4s…
            print(f"⚠️ {what} 临时失败（{type(e).__name__}），{delay:.0f}s 后进行第 {attempt} 次重试…")
            time.sleep(delay)
