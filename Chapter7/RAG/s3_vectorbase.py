'''设计向量数据库来存放文档和对应的向量表示
以及检索模块来实现根据Query来检索相关文档片段
主要功能包括：
    persist ：数据库持久化保存
    load_vector ：从本地加载数据库
    get_vector ：获取⽂档的向量表示
    query ：根据问题检索相关⽂档⽚段
'''


import os
from typing import Dict, List, Optional, Tuple, Union
import json
from Embeddings import BaseEmbeddings, OpenAIEmbedding
import numpy as np
from tqdm import tqdm


class VectorStore:
    def __init__(self, document: List[str] = ['']) -> None:
        # 存储原始的文本块列表（切分后的文档）
        self.document = document

    def get_vector(self, EmbeddingModel: BaseEmbeddings) -> List[List[float]]:
        
        self.vectors = []
        for doc in tqdm(self.document, desc="Calculating embeddings"):
            # 调用模型获取每个文本块的向量，并存入 self.vectors 列表
            self.vectors.append(EmbeddingModel.get_embedding(doc))
        return self.vectors

    def persist(self, path: str = 'storage'):
        if not os.path.exists(path):
            os.makedirs(path)

        # 保存原始文本到 document.json
        with open(f"{path}/doecment.json", 'w', encoding='utf-8') as f:
            json.dump(self.document, f, ensure_ascii=False)

        # 保存计算好的向量到 vectors.json
        if self.vectors:
            with open(f"{path}/vectors.json", 'w', encoding='utf-8') as f:
                json.dump(self.vectors, f)
    
    # 下次运行程序时，直接从硬盘读取，不需要重新调用 API 计算向量
    def load_vector(self, path: str = 'storage'):
        with open(f"{path}/vectors.json", 'r', encoding='utf-8') as f:
            self.vectors = json.load(f)
        with open(f"{path}/doecment.json", 'r', encoding='utf-8') as f:
            self.document = json.load(f)

    # 调用基类中定义好的余弦相似度静态工具来计算相似度
    def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        return BaseEmbeddings.cosine_similarity(vector1, vector2)

    def query(self, query: str, EmbeddingModel: BaseEmbeddings, k: int = 1) -> List[str]:
        # 1. 将用户的查询词（问题）转换成向量
        query_vector = EmbeddingModel.get_embedding(query)

        # 2. 计算问题向量与库中“所有”文档向量的相似度，得到一个分数数组
        result = np.array([self.get_similarity(query_vector, vector)
                          for vector in self.vectors])
        
        # 3. 排序并取前 k 个结果：
        # argsort() 返回从小到大的索引
        # [-k:] 截取分数最高的 k 个
        # [::-1] 倒序，让分数最高的排在最前面
        # 最后根据索引从 self.document 提取文本并转回列表
      
        '''
        argsort() 不会改变原数组，而是返回一个索引数组，表示数据从小到大排列时应该排在什么位置。
        例如：
        执行前：[0.1, 0.2, 0.9, 0.8]
        执行后：[0, 1, 3, 2]也就是最大的数的索引是2
        [-k:] 截取分数最高的 k 个,假设k=2,则返回[3,2]
        [::-1] 进行翻转,按照从大到小的顺序->[2,3]
        '''
        return np.array(self.document)[result.argsort()[-k:][::-1]].tolist()