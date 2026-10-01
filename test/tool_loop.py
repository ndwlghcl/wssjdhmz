# import re
#
# from get_current_time import get_current_time
# from llm import LLM
# from message import Message
# from tool_executor import ToolExecutor
# REACT_PROMPT_TEMPLATE = """
# 请注意，你是一个有能力调用外部工具的智能助手。
#
# 可用工具如下:
# {tools}
#
# 请严格按照以下格式进行回应:
#
# Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
# Action: 你决定采取的行动，必须是以下格式之一:
# - `{{tool_name}}[{{tool_input}}]`:调用一个可用工具。
# - `Finish[最终答案]`:当你认为已经获得最终答案时。
# - 当你收集到足够的信息，能够回答用户的最终问题时，你必须在Action:字段后使用 Finish[最终答案] 来输出最终答案。
#
# 现在，请开始解决以下问题:
# Question: {question}
# History: {history}
# """
#
# class ToolLoop:
#     def __init__(self):
#
#
#         pass
#
#     def _parse_action(self, action_text: str):
#         """解析Action字符串，提取工具名称和输入。
#         """
#         match = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL)
#         if match:
#             return match.group(1), match.group(2)
#         return None, None
#
#     def _parse_output(self, text: str):
#         """解析LLM的输出，提取Thought和Action。
#         """
#         # Thought: 匹配到 Action: 或文本末尾
#         thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
#         # Action: 匹配到文本末尾
#         action_match = re.search(r"Action:\s*(.*?)$", text, re.DOTALL)
#         thought = thought_match.group(1).strip() if thought_match else None
#         action = action_match.group(1).strip() if action_match else None
#         return thought, action
#
#     def loop(self):
#         llm=LLM()
#         my_tool_executor=ToolExecutor()
#         my_tool_executor.register_tool(get_current_time)
#
#         max_steps = 5
#         current_step=0
#         history = ''
#
#         while current_step<max_steps:
#             current_step+=1
#             #发消息给llm
#             prompt=REACT_PROMPT_TEMPLATE.format(tools=my_tool_executor.get_available_tools_str(),
#                                                 question="现在几点了",
#                                                 history=history)
#             message=Message(role="system",content=prompt)
#
#             #收到llm返回
#             response= llm.think([message.to_dict()])
#             #分析llm返回
#             thought, action = self._parse_output(response)
#
#             if thought:
#                 print(f"思考: {thought}")
#
#             if not action:
#                 print("警告:未能解析出有效的Action，流程终止。")
#                 break
#
#             # 4. 执行Action
#             if action.startswith("Finish"):
#                 # 如果是Finish指令，提取最终答案并结束
#                 finish_match = re.match(
#                     r"Finish\s*[\[【]\s*(.*?)\s*[\]】]",
#                     action,
#                     re.DOTALL
#                 )
#                 if finish_match:
#                     final_answer = finish_match.group(1)
#                 else:
#                     final_answer = action[len("Finish"):].strip(" :：[]【】")
#                 print(f"🎉 最终答案: {final_answer}")
#                 return final_answer
#
#             #是工具调用请求，分析请求
#             tool_name, tool_input = self._parse_action(action)
#             if not tool_name or not tool_input:
#                 # ... 处理无效Action格式 ...
#                 continue
#
#             print(f"🎬 行动: {tool_name}[{tool_input}]")
#
#             tool_function = my_tool_executor.get_tool(tool_name)
#             if not tool_function:
#                 observation = f"错误:未找到名为 '{tool_name}' 的工具。"
#             else:
#                 try:
#                     observation = my_tool_executor.execute_tool(tool_name)  # 调用真实工具
#                 except Exception as e:
#                     observation = f"错误:工具 '{tool_name}' 执行失败: {e}"
#
#             #执行工具
#
#             print(f"👀 观察: {observation}")
#
#             #返回工具调用结果
#
#             history=history + f"Thought: {thought}\nAction: {action}\nObservation: {observation}"
#
#         print("已达到最大步数，流程终止。")
#         return None
#
#     def run(self):
#         return self.loop()
#
# if __name__ == '__main__':
#     tool_loop=ToolLoop()
#     tool_loop.run()

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from get_current_time import get_current_time
from llm import LLM
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
    def __init__(self, max_steps: int = 5, temperature: float = 0.0):
        self.llm = LLM()
        self.tool_executor = ToolExecutor()
        self.tool_executor.register_tool(get_current_time)
        self.max_steps = max_steps
        self.temperature = temperature

    def _parse_action(self, action_text: str):
        """解析 Action 字符串，提取工具名称和输入。参数允许为空。"""
        cleaned = action_text.strip().strip("`").strip()
        match = re.search(r"([A-Za-z_]\w*)\s*\[(.*?)\]", cleaned, re.DOTALL)
        if match:
            return match.group(1), match.group(2).strip()
        return None, None

    def _parse_output(self, text: str):
        """解析 LLM 的输出，提取 Thought 和 Action。"""
        thought_match = re.search(r"Thought:\s*(.*?)(?=\n\s*Action:|$)", text, re.DOTALL)
        action_match = re.search(r"Action:\s*(.*)$", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _extract_final_answer(self, action: str) -> str:
        finish_match = re.search(r"Finish\s*[\[【]\s*(.*?)\s*[\]】]", action, re.DOTALL)
        if finish_match:
            return finish_match.group(1)
        idx = action.lower().find("finish")
        return action[idx + len("Finish"):].strip(" :：[]【】")

    def loop(self, question: str):
        trace: list[str] = []

        for step in range(1, self.max_steps + 1):
            print(f"\n--- 第 {step} 步 ---")

            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=self.tool_executor.get_available_tools_str(),
                question=question,
                history="\n\n".join(trace) if trace else "（暂无历史，这是第一步）",
            )
            messages = [{"role": "system", "content": prompt}]

            try:
                response = self.llm.think(messages, temperature=self.temperature)
            except Exception as e:
                print(f"[错误] 调用 LLM 失败: {e}")
                return f"抱歉，模型调用失败：{e}"

            thought, action = self._parse_output(response)
            if thought:
                print(f"[思考] {thought}")

            if not action:
                print("[警告] 未能解析出有效的 Action，已提示模型重试。")
                trace.append(response)
                trace.append("系统提示: 你的回复不符合格式，必须包含 "
                             "'Action: 工具名[参数]' 或 'Action: Finish[最终答案]'")
                continue

            if "finish" in action.lower():
                final_answer = self._extract_final_answer(action)
                print(f"[完成] 最终答案: {final_answer}")
                return final_answer

            tool_name, tool_input = self._parse_action(action)
            if not tool_name:
                print(f"[警告] Action 格式无法解析: {action}")
                trace.append(f"Thought: {thought or '（无）'}\nAction: {action}")
                trace.append("系统提示: Action 格式错误，正确写法是 工具名[参数] 或 Finish[最终答案]")
                continue

            print(f"[行动] {tool_name}[{tool_input}]")

            if tool_input:
                observation = self.tool_executor.execute_tool(tool_name, input=tool_input)
            else:
                observation = self.tool_executor.execute_tool(tool_name)

            print(f"[观察] {observation}")

            trace.append(
                f"Thought: {thought or '（无）'}\n"
                f"Action: {tool_name}[{tool_input}]\n"
                f"Observation: {observation}"
            )

        print("\n已达到最大步数，流程终止。")
        return "抱歉，我尝试了多次仍未能完成任务。"

    def run(self, question: str = "现在几点了"):
        return self.loop(question)


if __name__ == '__main__':
    tool_loop = ToolLoop()
    answer = tool_loop.run("现在几点了")
    print(f"\n最终答案: {answer}")
