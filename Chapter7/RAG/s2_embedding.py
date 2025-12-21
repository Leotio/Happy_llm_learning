'''将文本片段向量化，将一段文本映射为一个向量'''


import os
from copy import copy
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from openai import OpenAI

from dotenv import load_dotenv, find_dotenv


# find_dotenv(): 自动在当前的文件夹（以及父级文件夹）中寻找名为 .env 的文件，并返回它的完整路
# load_dotenv(...): 读取 .env 文件里的内容（比如 OPENAI_API_KEY=sk-xxxx），并把它们变成系统的环境变量

# _: _ 是一个惯例，表示运行这个函数，但并不关心它返回的具体结果
_ = load_dotenv(find_dotenv())


class BaseEmbeddings:
    def __init__(self, path: str, is_api: bool) -> None:
        self.path = path # 存储模型的路径或 API 的基础 URL/Key（如果是在线服务）
        self.is_api = is_api # True 代表走网络接口，False 代表加载本地显卡里的模型文件
    
    def get_embedding(self, text: str, model: str) -> List[float]:
        """
        获取文本的嵌入向量表示
            text (str): 输入文本
            model (str): 使用的模型名称
        Returns:
            文本的嵌入向量
        Raises:
            NotImplementedError: 该方法需要在子类中实现
        """
        # 在基类里先占位，所以model 只要是 BaseEmbeddings 的子类，就一定有 get_embedding

        # 这个方法在基类里是不写具体逻辑，子类继承父类的时候去实现这个方法
        # 如果不实现，那就还是在用父类的空客方法，就会报错！！！
        # 实现了的话就走子类自己的逻辑了
        raise NotImplementedError
    
    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> float:
        """
        计算两个向量之间的余弦相似度
            vector1 : 第一个向量
            vector2 : 第二个向量
        Returns:
            float: 两个向量的余弦相似度，范围在[-1,1]之间
        """
        # 将输入列表转换为numpy数组，并指定数据类型为float32
        v1 = np.array(vector1, dtype=np.float32)
        v2 = np.array(vector2, dtype=np.float32)

        # 检查向量中是否包含无穷大或NaN值
        # any要求括号里的所有元素都必须为 True才返回 True
        if not np.all(np.isfinite(v1)) or not np.all(np.isfinite(v2)):
            return 0.0

        # 计算向量的点积
        dot_product = np.dot(v1, v2)
        # 计算向量的范数（长度）
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        # 计算分母（两个向量范数的乘积）
        magnitude = norm_v1 * norm_v2
        # 处理分母为0的特殊情况
        if magnitude == 0:
            return 0.0
            
        # 返回余弦相似度
        return dot_product / magnitude
    
# BaseEmbeddings 基类的具体实现

class OpenAIEmbedding(BaseEmbeddings):

    def __init__(self, path: str = '', is_api: bool = True) -> None:
        # super().__init__：调用父类（基类）的初始化方法
        # 把 path 和 is_api 传给父类，完成基础属性的赋值
        super().__init__(path, is_api)

        '''连接程序与云端大模型服务的桥梁'''
        if self.is_api:
            # 实例化 OpenAI 客户端对象
            self.client = OpenAI()
            # 从操作系统的“环境变量”中读取OPENAI_API_KEY的值，并把它设为客户端的密钥
            self.client.api_key = os.getenv("OPENAI_API_KEY")
            # 从环境变量中获取 硅基流动 的基础URL
            self.client.base_url = os.getenv("OPENAI_BASE_URL")
    
    def get_embedding(self, text: str, model: str = "BAAI/bge-m3") -> List[float]:
        """
        此处默认使用轨迹流动的免费嵌入模型 BAAI/bge-m3
        """
        if self.is_api:
            # 文本清洗：将换行符替换为空格
            # 因为有些 Embedding 模型对换行符敏感，去掉换行符通常能获得更稳定的向量
            text = text.replace("\n", " ")

            # 调用 SDK 接口：
            # 1. input=[text]：把文本送进去
            # 2. model=model：指定使用的模型
            # 3. .data[0].embedding：从返回的复杂 JSON 结果中层层提取，只拿取最终的 float 列表（向量）
            return self.client.embeddings.create(input=[text], model=model).data[0].embedding
       
        else:
            # 如果 is_api 为 False，说明用户想用本地模型
            # 但当前类还没写本地模型的加载逻辑，所以抛出错误
            raise NotImplementedError