'''Embedding层的实现'''
# 将自然语言输入转化为机器可以处理的向量
'''
Embedding 层的输⼊往往是⼀个形状为 (batch_size, seq_len, 1)的矩阵:
    第⼀个维度是⼀次批处理的数量，
    第⼆个维度是⾃然语⾔序列的⻓度，
    第三个维度则是 token 经过 tokenizer 转化成的 index 值(指示该token在原句子的索引位置)
'''
# 直接调用torch中的Embedding层即可实现
import torch.nn as nn
# Embedding层内部维护一个可训练的权重矩阵W_emb：args.vocab_size,args.dim
# 对于每一个输入的索引i，Output=W_emb[i]，不需要进行矩阵乘法，只需要根据输入的索引i查找矩阵的第 i 行
self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)