from types import SimpleNamespace

from ReAct_agent import ReActAgent


class FakeLLM:
    def chat(self, messages, **kwargs):
        return SimpleNamespace(
            content="这是预设的回答",
            tool_calls=[],
        )


agent = ReActAgent(name="测试助手", llm=FakeLLM())
answer = agent.run("你好")

assert answer == "这是预设的回答"
assert agent.get_history()[-1] == {
    "role": "assistant",
    "content": "这是预设的回答",
}

print("测试通过")

class FakeTimeLLM:
    def __init__(self):
        self.calls = 0

    def chat(self, messages, **kwargs):
        self.calls += 1

        if self.calls == 1:
            return SimpleNamespace(
                content=None,
                tool_calls=[
                    SimpleNamespace(
                        id="call_time_001",
                        function=SimpleNamespace(
                            name="get_current_time",
                            arguments="{}",
                        ),
                    )
                ],
            )

        # 第二次调用时，模型应该已经收到工具结果
        observation = messages[-1]
        assert observation["role"] == "tool"
        assert observation["tool_call_id"] == "call_time_001"

        return SimpleNamespace(
            content="工具返回的时间：" + observation["content"],
            tool_calls=[],
        )


fake = FakeTimeLLM()
agent = ReActAgent(name="工具测试助手", llm=fake)
answer = agent.run("现在几点？")

assert fake.calls == 2
assert [m["role"] for m in agent.get_history()] == [
    "system", "user", "assistant", "tool", "assistant",
]
assert answer.startswith("工具返回的时间：")

print("工具调用测试通过")