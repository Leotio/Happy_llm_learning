'''Attention : GQA实现'''
import torch
import torch.nn as nn
from typing import Tuple
from s1_modelconfig import ModelConfig
import math
import torch.nn.functional as F

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
def precompute_freqs_cis(dim: int,end: int, theta: float = 10000.0):# end为序列长度
    # freqs也就是θ_i， 为10000 ^ (-2i/d) = (1/10000) ^ (2i/d)  i是维度分组索引
    # torch.arange(0,dim,2):从0开始，到dim，步长为2
    # [:(dim//2)]确保生成的序列的长度为dim的一半
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    # t为从0开始到end的序列（公式里面的m）
    t = torch.arange(end, device=freqs.device)
    # 计算外积,得到m*θ_i
    freqs = torch.outer(t, freqs).float()
    # 计算频率的余弦值，得到实部
    freqs_cos = torch.cos(freqs)
    # 计算频率的正弦值，得到虚部
    freqs_sin = torch.sin(freqs)

    # freqs_cos, freqs_sin都是[end, dim // 2]
    return freqs_cos, freqs_sin 

# 用于调整张量形状，使之与x的维度对齐
# x是词向量经过 Q、K 变换后，强制改造成复数结构，两两一组，维度[Batch_Size, Seq_Len, n_heads, head_dim // 2]
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
    # xq为(batch_size, seq_len, N_heads, dim)已经经过了wk，wq矩阵的变换
    # xq.shape[:-1]为(batch_size, seq_len, N_heads)
    # + (-1, 2)则意味着重塑为(batch_size, seq_len, N_heads，-1，2)意味着将dim分为dim//2 * 2
    # 即(batch_size, seq_len, N_heads，dim//2，2)
    # 把dim维度上的元素分为 两个一组 
    ''' 
    .unbind(-1): 在最后一个维度 (大小为 2) 上解包得到：
        xq_r: 包含每个二维子向量的第一个元素(视为实部)
        xq_i: 包含每个二维子向量的第二个元素(视为虚部)'''
    # xq_r，xk_r拿走每组的第一个数 
    # xq_i，xk_i拿走每组的第二个数 
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
    # xq_r 每组的第一个数 ，xq_i 每组的第二个数
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


'''RoPE测试代码'''
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
        # 根据是否指定了K、V头数来确定key和value头
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # 确保总头数可以被键值头数整除
        assert args.n_heads % self.n_kv_heads == 0
        # 模型并行处理大小，默认为1
        model_parallel_size = 1
        # 本地计算头数，等于总头数除以模型并⾏处理⼤⼩
        self.n_local_heads = args.n_heads // model_parallel_size
        # 本地键值头数，等于键值头数除以模型并⾏处理⼤⼩
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        # 重复次数，⽤于扩展键和值的尺⼨
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # 每个头的维度，等于模型维度除以头的总数
        self.head_dim = args.dim // args.n_heads

        # 定义权重矩阵
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        # 输出权重矩阵
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # 定义dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        # 保存dropout概率
        self.dropout = args.dropout

        # 检查是否使用Flash Attention
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            # 不支持Flash Attention时，使用手动实现注意力机制，并设置mask
            # Flash Attention提供了快捷、高效、内建Mask的路径；没有则必须走手动、低效的路径，并需要手动创建和应用因果Mask
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # 创建上三角mask
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            # 注册为模型的缓冲区
            self.register_buffer("mask", mask)
    def forward(self, x: torch.Tensor, freq_cos: torch.Tensor, freq_sin: torch.Tensor):
        # 获取batchsize和序列长度
        # 初始输入的x维度应该是（batchsize，seqlen，dim）
        bsz, seqlen, _ = x.shape

        # 计算Q，K，V
        # xq为(bsz, seqlen, n_heads * head_dim)
        # xk为(bsz, seqlen, n_kv_heads * head_dim)
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # 调整形状适应头的维度
        # view操作将最后一个维度拆解为两个更小的维度
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # 应用旋转位置嵌入
        xq, xk = apply_rotary_emb(xq, xk, freq_cos, freq_sin)

        # 对k和v头进行repeat来符合q头
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # 将头作为超级批次处理
        # (B,S,H,d_h)->(B,H,S,d_h)
        xq = xq.transpose(1,2)
        xk = xk.transpose(1,2)
        xv = xv.transpose(1,2)

        # 根据是否支持Flash Attention来实现注意力计算
        if self.flash:
            # 使用Flash Attention
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv,
                attn_mask=None, # 注意力掩码,通常将其设置为None,如果需要应用非因果或自定义Mask才设置为True
                # 训练时,使用 self.dropout 设定的概率;推理时,使用 0.0,即关闭Dropout
                # 该参数控制在 Softmax 之后、与 V 相乘之前，应用于注意力得分矩阵的 Dropout 概率。
                dropout_p=self.dropout if self.training else 0.0, 
                is_causal=True,
            )
        else:
            # 手动实现注意力机制
            # 注意力是在每个独立的头Head内部计算的，所以使用的放缩的维度是self.head_dim
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            # self 对象是否拥有名为 'mask' 的属性
            assert hasattr(self, 'mask')
            scores = scores + self.mask[:, :, :seqlen, :seqlen]
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)

       
        # transpose:(bsz,N_heads,seqlen,d_h)->(bsz,seqlen,N_heads,d_h)把之前为了合成超级batch进行的tranpose换回来
        # contiguous()强制 PyTorch 在内存中重新排列张量，使其恢复连续存储
        # view()操作要求张量内存必须是连续的,在执行view之前调用contiguous()是常见的
        #  将(N_heads,d_h)两个维度合并成总的特征维度
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)

        # 经过w0投影进行特征混合一下
        output = self.wo(output)
        # 注意力输出与残差连接相加之前，应用 Dropout 进行正则化
        output = self.resid_dropout(output)

        # output代表了经过注意力机制处理后，每个 Token 包含的最新、最丰富的上下文信息
        return output

'''Attention测试代码'''
# 创建Attention实例
attention_model = Attention(args)
# 模拟输⼊数据
batch_size = 1
seq_len = 50 # 假设实际使⽤的序列⻓度为50
dim = args.dim
x = torch.rand(batch_size, seq_len, dim) # 随机⽣成输⼊张量
# freqs_cos = torch.rand(seq_len, dim // 2) # 模拟cos频率，⽤于RoPE
# freqs_sin = torch.rand(seq_len, dim // 2) # 模拟sin频率，⽤于RoPE
freqs_cos, freqs_sin = precompute_freqs_cis(dim//args.n_heads, seq_len)
# 运⾏Attention模型
output = attention_model(x, freqs_cos, freqs_sin)
# attention出来之后的形状 依然是[batch_size, seq_len, dim]
print("Output shape:", output.shape)