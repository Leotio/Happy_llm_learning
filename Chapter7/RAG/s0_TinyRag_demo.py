from s3_vectorbase import VectorStore
from s1_utils import ReadFiles
from s4_baseLLM import OpenAIChat
from s2_embedding import OpenAIEmbedding

# 获得data目录下的所有文件内容并分割
docs = ReadFiles('./data').get_content(max_token_len=600, cover_content=150) 
# 创建向量存储对象，存储文档块
vector = VectorStore(docs)

# 创建EmbeddingModel嵌入模型
embedding = OpenAIEmbedding() 
# 使用嵌入模型将所有文档块转换为向量
vector.get_vector(EmbeddingModel=embedding)

# 将向量和文档内容保存到storage目录下，下次再用就可以直接加载本地的数据库
vector.persist(path='storage') 

# vector.load_vector('./storage') # 加载本地的数据库

question = 'RAG的原理是什么？'

# 返回最相关的k(1)个文档块,[0]: 取结果列表的第一个（也是唯一一个）结果
content = vector.query(question, EmbeddingModel=embedding, k=1)[0]

# 创建大语言模型实例，使用通义千问的32B版本
chat = OpenAIChat(model='Qwen/Qwen2.5-32B-Instruct')

#  question:用户的问题;[]: 历史对话记录（这里为空）;content: 检索到的相关文档内容
print(chat.chat(question, [], content))