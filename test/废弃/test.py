import re

from get_current_time import get_current_time
from llm import LLM
from message import Message
from tool_executor import ToolExecutor
REACT_PROMPT_TEMPLATE = """请注意，你是一个有能力调用外部工具的智能助手。

## 可用工具
{tools}

## 输出格式（必须严格遵守）
Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下两种格式之一:
- `工具名[参数]`：调用一个可用工具；若该工具无需参数，写成 `工具名[]`
- `Finish[最终答案]`：当你已经获得足够信息、可以回答用户问题时使用

## 重要提醒
1. 每次回应只能包含一组 Thought + Action，禁止一次输出多步
2. 严禁自己编造 Observation，工具的执行结果由系统返回给你
3. Action 必须用方括号包裹参数，不要添加多余的解释文字

## 当前任务
Question: {question}

## 执行历史
{history}

现在开始你的推理和行动:
"""

class ToolLoop:
    def __init__(self):


        pass

    def _parse_action(self, action_text: str):
        """解析Action字符串，提取工具名称和输入。
        """
        match = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL)
        if match:
            return match.group(1), match.group(2)
        return None, None

    def _parse_output(self, text: str):
        """解析LLM的输出，提取Thought和Action。
        """
        # Thought: 匹配到 Action: 或文本末尾
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
        # Action: 匹配到文本末尾
        action_match = re.search(r"Action:\s*(.*?)$", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def loop(self):
        llm=LLM()
        my_tool_executor=ToolExecutor()
        my_tool_executor.register_tool(get_current_time)

        max_steps = 5
        current_step=0
        history = ''

        while current_step<max_steps:
            current_step+=1
            #发消息给llm
            prompt=REACT_PROMPT_TEMPLATE.format(tools=my_tool_executor.get_available_tools_str(),
                                                question="现在几点了",
                                                history=history)
            message=Message(role="system",content=prompt)

            #收到llm返回
            response= llm.think([message.to_dict()])
            #分析llm返回
            thought, action = self._parse_output(response)

            if thought:
                print(f"思考: {thought}")

            if not action:
                print("警告:未能解析出有效的Action，流程终止。")
                break

            # 执行Action
            if action.startswith("Finish"):
                # 如果是Finish指令，提取最终答案并结束
                finish_match = re.match(
                    r"Finish\s*[\[【]\s*(.*?)\s*[\]】]",
                    action,
                    re.DOTALL
                )
                if finish_match:
                    final_answer = finish_match.group(1)
                else:
                    final_answer = action[len("Finish"):].strip(" :：[]【】")
                print(f"🎉 最终答案: {final_answer}")
                return final_answer

            # 执行工具调用请求
            tool_name, tool_input = self._parse_action(action)
            if not tool_name:
                # 格式无效时给模型一个纠错提示，否则下一轮它会重复同样的输出
                history += "系统提示: Action 格式无效，请使用 `工具名[参数]` 或 `Finish[最终答案]`\n"
                continue

            print(f"🎬 行动: {tool_name}[{tool_input}]")

            # execute_tool 内部已处理"工具不存在"和"执行报错"，外面不用再查一遍
            # 注意: 工具需要参数时才传，get_current_time 不接收参数
            if tool_input:
                observation = my_tool_executor.execute_tool(tool_name, input=tool_input)
            else:
                observation = my_tool_executor.execute_tool(tool_name)

            print(f"👀 观察: {observation}")

            # 返回工具调用结果（注意结尾的 \n，否则下一轮的 Thought 会黏在上一轮的 Observation 后面）
            history += f"Thought: {thought or ''}\nAction: {action}\nObservation: {observation}\n"

        print("已达到最大步数，流程终止。")
        return None

    def run(self):
        return self.loop()

if __name__ == '__main__':
    tool_loop=ToolLoop()
    tool_loop.run()