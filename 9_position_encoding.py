''' PE: 位置编码层的实现 '''
# PE 的定义：绝对位置，与内容无关,所以求解出来就可以用于所有样本的位置编码！
# 计算公式见pe_format.png

import torch
import torch.nn as nn
import math
class PositionalEncoding(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Dropout层
        # 原始Transformer架构在PE后是接上了Dropout层的，用于正则化固定PE信息和防止模型过拟合
        self.dropout = nn.Dropout(p=args.dropout)

        # block size是序列的最大长度
        # 创建这样shape的pe，因为求出值后，需要与原始的词嵌入结果相加
        pe = torch.zeros(args.block_size, args.embd)
        # torch.arange(start, end, step)不包含end
        # unsqueeze(1)：增加一个维度，索引 0 代表在最前面插入，索引 1 代表在第一个维度后面插入
        # 将张量的形状从L_max变为(L_max, 1)，为后续与 div_term 矩阵的广播相乘做好了准备。
        position = torch.arange(0, args.block_size).unsqueeze(1)

        # 计算theta(即sin和cos的角度)
        # div_term:分母项,div_term = 1 / 10000^{2i / d_model}
        '''
        torch.arange(0, args.embd, 2)生成包含所有偶数索引的张量
        -(math.log(10000.0) / args.embd: -ln(10000)/d_model
        由exp(a *ln b) = b^a:
        故: exp( (-2i*ln10000) / d_model ) = 10000 ^ (-2i/d_model)
        '''
        div_term = torch.exp(
            torch.arange(0, args.embd, 2) * -(math.log(10000.0) / args.embd)
        )

        # 分别计算sin和cos项
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # (block_size, n_embd) -> (1, block_size, n_embd)方便后续在前向传播中进行广播
        pe = pe.unsqueeze(0)
        # 注册到缓存区
        self.register_buffer("pe", pe)

    def forward(self, x):
        # 将位置编码加到Embedding结果上
        # x:(B, L', D),pe需要切片为（1，L',D)
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return x