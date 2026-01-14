'''实现基本的注意力实现 + 自注意力 + 掩码自注意力'''

'''基本注意力机制'''
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
    # transpose(-2,-1)将key矩阵的倒数第二个维度、倒数第一个维度交换
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

''' 自注意力机制：'''
# KQV都是同一个矩阵就欧克
x = torch.tensor([[1, 2, 3], [4, 5, 6]]) #随意生成的矩阵x
attention(x, x, x)

''' 掩码自注意力:'''
# 创建一个上三角矩阵，用于遮蔽信息
# 假设这里的输入维度是(batch_size, seq_len, hidden_size) hideen_size是embedding模型生成的向量的维度
# full()函数创建一个1 * max_seq_len * max_seq_len的矩阵，其值全部为-inf
# 注意此处max_seq_len是模型能处理的最大序列长度，后续进行计算的时候需要进行动态切片
# 这个1主要是为了方便进行广播机制
mask = torch.full((1, args.max_seq_len, args.max_seq_len), float('-inf'))

# triu(input, diagonal)保留包括diagonal这条对角线及其之上的元素，其余的变为0
# diagonal=0代表主对角线，1则代表主对角线上边一条对角线
mask = torch.triu(mask, diagonal=1)

# 保留mask的第一个维度
# 但是要将 mask 的最后两个维度（通常是序列长度维度）都切片到 seqlen 的大小
# 这确保了 mask 的大小与 scores 的最后两个维度 (T_q, T_k) 匹配，以便进行逐元素相加。
# 隐含了广播操作：如果 scores是 4D(B, H, T, T)，而 mask是3D(B, T, T)，PyTorch会自动在H维度上广播。
scores = scores + mask[:, :seqlen, :seqlen] 

# 在scores的最后一个维度上进行softmax操作，
# 最后一个维度是T_k（原序列的长度），让每个query对原序列的掩码注意力之和为1
# float()很有必要：Softmax涉及指数运算，高精度可以防止数值溢出和计算不稳定
# type_as(xq)将因为float改变的精度修改回去
# xq: 输入查询张量
scores = F.softmax(scores.float(), dim=-1).type_as(xq)