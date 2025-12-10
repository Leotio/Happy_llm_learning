'''Attention : GQA实现'''
import torch
import torch.nn as nn
from typing import Tuple
from s1_modelconfig import ModelConfig

'''扩展K和V的维度到Q的维度'''
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # batch_size, 序列长度，key/value头的数量、每个头的维度大小
    bs, slen, n_kv_heads, head_dim = x.shape

    # 重复次数为1直接返回原始张量
    if n_rep == 1:
        return x
    
    return(
        x[:, :, :, None, :] # 在第四个维度前添加一个新的维度
        .expand(bs, slen, n_kv_heads, n_rep, head_dim) # 新的维度扩展到n_rep大小，也就是对其进行一个重复
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim) # 合并key/value头的数量和重复次数的维度
    )# 最终使张量形状达到与查询的维度一致的效果

'''旋转嵌入Rotary Position Embeddings, RoPE'''

# 获得旋转嵌入的实部和虚部
def precompute_freqs_cis(dim: int,end: int, theta: float = 10000.0):
    # freqs(θ_i) = 10000 ^ (-2i/d) = (1/10000) ^ (2i/d)  i是维度分组索引
    # torch.arange(0,dim,2):从0开始，到dim，步长为2
    # [:(dim//2)]确保生成的序列的长度为dim的一半
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    # 从0开始到end的序列
    t = torch.arange(end, device=freqs.device)
    # 计算外积,得到m*θ_i
    freqs = torch.outer(t, freqs).float()
    # 计算频率的余弦值，得到实部
    freqs_cos = torch.cos(freqs)
    # 计算频率的正弦值，得到虚部
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin 

# 用于调整张量形状，使之与x的维度对齐
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    # 获取x的维度
    ndim = x.ndim

    # 确保1在x的维度范围内
    assert 0 <= 1 < ndim

    # 确保的freqs_cis形状与x的第二维和最后一维相同
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])

    # 构造新的形状，除第二维和最后一维，其余均为1，便于广播
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

# 实现最终的旋转嵌入
def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    # 查询和键张量转化为浮点数，并重塑形状分离实部和虚部
    # xq为(batch_size, seq_len, N_heads, dim)
    # xq.shape[:-1]为(batch_size, seq_len, N_heads)
    # + (-1, 2)则意味着重塑为(batch_size, seq_len, N_heads，-1，2)意味着将dim分为dim//2 * 2
    # 即(batch_size, seq_len, N_heads，dim//2，2)
    ''' 
    .unbind(-1): 在最后一个维度 (大小为 2) 上解包得到：
        xq_r: 包含每个二维子向量的第一个元素(视为实部)
        xq_i: 包含每个二维子向量的第二个元素(视为虚部)'''
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # 重塑频率张量进行广播
    # freqs_cos为(seq_len,dim//2), xq_r为(batch_size, seq_len, N_heads，dim//2)
    # 将freqs_cos重塑为(1,seq_len,1,dim//2)
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # 应用旋转，得到旋转后的实部和虚部
    #  a+bi = (acosθ - bsinθ) + (asinθ + bcosθ)i
    # xq_out_r维度为(batch_size, seq_len, N_heads，dim//2)
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # 将最后两个维度合并，并还原为原始张量的形状
    # stack堆叠后，维度变为(batch_size, seq_len, N_heads，dim//2，2)
    # flatten(3)从第三个维度开始展平
    # 最终xq_out维度为(batch_size, seq_len, N_heads，dim)
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

'''测试代码'''
xq = torch.randn(1, 50, 6, 48) # bs, seq_len, dim//n_head, n_head_dim
xk = torch.randn(1, 50, 6, 48) # bs, seq_len, dim//n_head, n_head_dim
# 使⽤ precompute_freqs_cis 函数获取 sin和cos
cos, sin = precompute_freqs_cis(288//6, 50)
print(cos.shape, sin.shape)
xq_out, xk_out = apply_rotary_emb(xq, xk, cos, sin)
print(xq_out.shape, xk_out.shape)

'''构建LLaMA2 Attention模块'''
class Attention(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
