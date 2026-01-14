import torch
import torch.nn as nn
import math
import torch.functional as F

'''多头自注意力模块'''
class MultiHeadAttention(nn.Module):
    # ModelArgs集中管理构建模型所需的所有超参数
    # is_causal: 
    # False (默认值)双向注意力,模型在计算一个位置的输出时，可以访问整个序列
    # True	单向注意力,通过一个上三角掩码（Upper Triangular Mask）实现
    def __init__(self, args: ModelArgs, is_casual=False):
        super().__init__()
        # 隐藏层维度（其实就是embedding的维度）必须是头数的整数倍，按照多头注意力的实现，我们需要把输入拆解成头数个矩阵
        # 断言不满足会抛出异常，直接返回
        assert args.dim % args.n_heads == 0
        # 模型并行处理的大小，也就是设备的数量，代表放到几个GPU上面去运行
        model_parallel_size = 1
        # 代表此设备需要计算的头数
        self.n_local_heads = args.n_heads // model_parallel_size
        # 每个头的维度，是模型的总维度除以头数
        self.head_dim = args.dim // args.n_heads

        # 参数矩阵，大小为n_embd × n_embd
        # nn.Linear(输入维度, 输出维度, bias)用于实现全连接/线性变换的模块
        # 线性层实现y = x W^T + b，所以输入维度=W的列数；输出维度=W的行数
        # 所以此处的.weight矩阵应该是(args.n_local_head * self.head_dim,args.dim) 的
        # 通过这个nn.Linear来定义输入输出的维度，使得其是可学习的参数，他会在这一步随机初始化一个W矩阵！
        # 此处定义的wq这些都是对象，通过创建对象的时候Linear隐式创建的weight参数才是真正意义上的wq!
        # 后续调用的时候只需要传入input数据，就会和初始化的W矩阵计算后输出output数据
        self.wq = nn.Linear(args.dim, args.n_local_head * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_local_head * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_local_head * self.head_dim, bias=False)

        # 所有注意力头拼接起来，与wo相乘
        # wo的维度:输入是上面层的输出，所以input维度应该是args.n_local_head * self.head_dim
        # wo此处的输出必须是原始模型维度，也就是args.dim

        self.wo = nn.Linear(args.n_local_head * self.head_dim, args.dim, bias=False)

        # dropout必须使用 nn.Dropout实现，即使他是不可学习的参数
        # 因为只有当nn.Dropout 继承自 nn.Module时才能获得两个关键方法：
            # model.train()：nn.Dropout 会进入训练模式，执行随机丢弃。
            # model.eval()：nn.Dropout 会进入评估模式，不执行丢弃。
        # 注意力的dropout
        # Softmax激活之后，与V矩阵相乘之前的输出之后
        self.attn_dropout = nn.Dropout(args.dropout)
        # 残差连接的dropout
        # 在每个MHA或FFN的输出和残差连接相加之前，随机地丢弃子层（MHA、FFN）的输出
        self.resid_dropout = nn.Dropout(args.dropout)

        # 掩码注意力；因为为多头注意力，维度需要比之前多1
        if is_casual:
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float('-inf'))
            mask = torch.triu(mask, diagonal=1)
            # 注册到缓冲区，不会进行梯度更新；但是成为了模型状态的一部分
            self.register_buffer("mask", mask)

    def forward(self, q: torch.tensor, k: torch.tensor, v: torch.tensor):
        # 获取batch_size和序列长度，q[batch_size,seqlen,dim]
        # 这里的q,k,v实际上是还没有经过wq，wk，wv的变换的，相当于原始输入X
        bsz, seqlen, _ = q.shape

        # 计算真正的KQV,维度变换为(B, T, n_embed) x (n_embed, n_embed) -> (B, T, n_embed)
        xq, xk, xv = self.wq(q), self.wk(k), self.wv(v)

        # 将KQV拆分为多头(B,T,n_head,C//n_head)
        # view()不改变张量,只提供了新的视角（View）来看数据,但要求原始张量在内存中必须是连续存储
        # 此处则是将最后一维拆解为了self.n_local_heads*self.head_dim
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        # 将张量从(B, L, h, d_k)转置到(B, h, L, d_k),可以实现GPU的最大并行化
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 注意力计算，Q@K_T(B, n_h, T, h_dim) x (B, n_h, h_dim, T) -> (B, n_h, T, T)
        scores = torch.matmul(xq, xk.transpose(-1,-2)) / math.sqrt(self.head_dim)
        
        # 如果为掩码注意力，要在softmax前进行掩码操作、
        if self.is_causal:
            # hasattr（）检查对象是否有某个属性/方法
            assert hasattr(self, 'mask')
            scores = scores + self.mask[:, :, :seqlen, : :seqlen]

        # softmax操作
        scores = F.softmax(scores.float(), dim=-1).typeas(xq)
        
        # 第一次dropout操作
        scores = self.attn_dropout(scores)

        # scores与V相乘:(B, n_h, T, T) x (B, n_h, T, h_dim) -> (B, n_h, T, h_dim)
        output = torch.matmul(scores, xv)

        # output先把维度交换回去，再进行拼接操作，到正确的(B,L,d_k)
        # 交换维度为 (B, T, n_head, C // n_head)，再拼接成 (B, T, n_head * C // n_head)
        # contiguous 函数⽤于重新开辟⼀块新内存存储，因为Pytorch设置先transpose再view会报错，
        # 因为view直接基于底层存储得到，然⽽transpose并不会改变底层存储，只改变访问的步长，因此需要额外存储
        output = output.transpose(2, 3).contiguous().view(bsz, seqlen, -1)

        # 和wo进行计算，汇总各个头的信息
        output = self.wo(output)
        # 进行第二次的dropout
        output = self.resid_dropout(output)
        return output
