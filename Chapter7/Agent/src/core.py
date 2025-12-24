"""
定义 Agent 类,负责管理对话历史、调⽤ OpenAI API、处理⼯具调⽤请求以及执⾏⼯具函数
"""

from openai import OpenAI
import json
from typing import List, Dict, Any
from src.utils import function_to_json
from src.tools import get_current_datetime, add, compare, count_letter_in_string, search_wikipedia, get_current_temperature

import pprint

SYSTEM_PROMPT = """
你是一个叫不要葱姜蒜的人工智能助手。你的输出应该与用户的语言保持一致。
当用户的问题需要调用工具时，你可以从提供的工具列表中调用适当的工具函数。
"""

class Agent:
    def __init__(self, client: OpenAI, model: str = "Qwen/Qwen2.5-32B-Instruct", tools: List=[], verbose : bool = True):
        """
        Docstring for __init__
        client：接收一个已经配置好的 OpenAI 客户端实例
        model：指定使用的模型名称。代码中默认使用了 Qwen（通义千问）2.5 系列模型
        tools：接收一个工具函数列表
        verbose：布尔值。如果为 True，通常用于在运行过程中打印详细的中间日志
        """
        self.client = client
        self.tools = tools
        self.model = model
        # self.messages列表，保存所有对话记录，初始化时，把SYSTEM_PROMPT放入列表中
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        self.verbose = verbose

    # 获取所有工具的 JSON 模式
    def get_tool_schema(self) -> List[Dict[str, Any]]:
        
        return [function_to_json(tool) for tool in self.tools]
    

    # 处理工具调用
    def handle_tool_call(self, tool_call):
        # 模型想调用的函数名
        function_name = tool_call.function.name
        # 模型给出的参数（字符串格式的 JSON）
        function_args = tool_call.function.arguments
        # 调用的唯一标识符
        function_id = tool_call.id

        # 使用 eval 动态执行函数,
        # {function_args} 是 JSON 字符串，eval 结合语法将其解包为 Python 函数的关键字参数
        # eval() 的作用是：把字符串当成真实的 Python 代码来运行,所以这里就返回了tools调用的结果了
        function_call_content = eval(f"{function_name}(**{function_args})")


        # 返回符合 OpenAI 规范的工具执行结果格式
        return {
            "role": "tool",
            "content": function_call_content,
            "tool_call_id": function_id,
        }

    def get_completion(self, prompt) -> str:

        # 存到历史记录里头
        self.messages.append({"role": "user", "content": prompt})

        # 调用模型 API，获取初步响应
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.get_tool_schema(),
            stream=False,
        )

        # 判断是否需要调用tools
        if response.choices[0].message.tool_calls:
            # 将包含 tool_calls 的完整 assistant 消息添加到历史中
            assistant_message = {
                "role": "assistant",
                "content": response.choices[0].message.content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    }
                    for tool_call in response.choices[0].message.tool_calls
                ]
            }
            self.messages.append(assistant_message)

            # 初始化一个临时列表，用于在控制台打印日志
            tool_list = []

            # 遍历模型要求的所有工具调用
            for tool_call in response.choices[0].message.tool_calls:
                # 处理工具调用并将结果添加到消息列表中
                self.messages.append(self.handle_tool_call(tool_call))

                tool_list.append([tool_call.function.name, tool_call.function.arguments])
            
            if self.verbose:
                print("调用工具：", response.choices[0].message.content, tool_list)

            # 再次调用模型API，这次messages列表里包含了 用户问题+模型的调用指令+函数运行后的真实数据
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.get_tool_schema(),
                stream=False,
            )

        # 将模型的完成响应添加到消息列表中
        self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
        
        # 将最终的文本答案返回给调用者
        return response.choices[0].message.content


    
