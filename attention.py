'''实现基本的注意力实现 + 自注意力 + 掩码自注意力'''

# Scaled Dot-Product Attention缩放点积注意力实现
import torch
import math
def attention(query, key, value, dropout=None):
    '''
    args:
    query:查询值矩阵
    key:键值矩阵
    value:真值矩阵
    '''
    # 获取键值向量的维度，size(-1)返回键值矩阵（Batch, T_k, d_k）最后一个维度d_k
    d_k = query.size(-1)
    # 计算Q与K的内积并除根号d_k
    # 根号d_k：防止点积结果过大，导致后续 Softmax 函数的梯度变得极小（进入梯度饱和区）
    # trasnpose(-2,-1)将key矩阵的倒数第二个维度、倒数第一个维度交换
    scores = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(d_k)
    # Softmax操作（p_attn:probability attention）
    # softmax操作要在scores的最后一个维度上进行，
    # 最后一个维度是T_k（原序列的长度），让每个query对原序列的注意力之和为1
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 使用p_attn对value进行加权求和
    # 并和p_attn作为一个元组返回
    return torch.matmul(p_attn, value), p_attn

# 自注意力机制：
# KQV都是同一个矩阵就欧克
x = torch.tensor([[1, 2, 3], [4, 5, 6]]) #随意生成的矩阵x
attention(x, x, x)

# 掩码自注意力
