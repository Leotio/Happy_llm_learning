'''大模型模块，用于根据检索到的文档来回答用户的Query'''


import os
from typing import Dict, List, Optional, Tuple, Union
from openai import OpenAI

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


# RAG_PROMPT_TEMPLATE提示词工程（Prompt Engineering）的模板
# 作用：给模型立了规矩：1. 只能根据给定的上下文回答；2. 不知道不能瞎编；3. 用中文
# 占位符：{question} 和 {context} 是变量，程序运行时会把真实的问题和检索到的文档填进去
RAG_PROMPT_TEMPLATE="""
使用以上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
问题: {question}
可参考的上下文：
···
{context}
···
如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
有用的回答:
"""


class BaseModel:
    def __init__(self, model) -> None:
        self.model = model

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        pass # 占位，待子类实现具体逻辑

    def load_model(self):
        pass # 占位，如果是本地模型需要这个加载动作

# 利用 OpenAI SDK 协议调用大模型
class OpenAIChat(BaseModel):
    def __init__(self, model: str = "Qwen/Qwen2.5-32B-Instruct") -> None:
        self.model = model

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        # 1. 初始化客户端（同样从环境变量读取 Key 和 URL）
        client = OpenAI()
        client.api_key = os.getenv("OPENAI_API_KEY")   
        client.base_url = os.getenv("OPENAI_BASE_URL")

        # 2. 注入上下文：
        # 使用模板的 .format() 方法，把用户问题(prompt)和检索到的资料(content)嵌入模板
        # 然后把这个大字符串包装成 'user' 的角色，加入对话历史 history 中
        history.append({'role': 'user', 'content': RAG_PROMPT_TEMPLATE.format(question=prompt, context=content)})
        
        # 3. 发送请求：
        response = client.chat.completions.create(
            model=self.model,
            messages=history,
            max_tokens=2048,
            temperature=0.1
        )

        # 4. 提取回答：从返回的复杂对象中拿出文本内容
        '''
        resonse示例
        {
        "id": "chatcmpl-123",
        "choices": [
            {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "根据上下文，马里奥在吃到蘑菇后会变大。" 
            },
            "finish_reason": "stop"
            }
        ],
        "usage": { "total_tokens": 50 }
        }
        '''
        return response.choices[0].message.content