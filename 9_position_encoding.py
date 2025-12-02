''' PE: 位置编码层的实现 '''

import torch
import torch.nn as nn
class PositionalEncoding(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Dropout层
        # 原始Transformer架构在PE后是接上了Dropout层的，用于正则化固定PE信息和防止模型过拟合
        self.dropout = nn.Dropout(p=args.dropout)

        # block size是序列的最大长度
        # 创建这样shape的pe，因为求出值后，需要与原始的词嵌入结果相加
        pe = torch.zeros(args.block_size, args.embd)

    def forward(self, x):
        # 将位置编码加到Embedding结果上
        x = x + self.pe[]
        return x