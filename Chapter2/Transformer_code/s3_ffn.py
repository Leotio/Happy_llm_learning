'''前馈神经网络'''
# 由全连接层+激活函数构成，MLP多层感知机是最简单的FFN

# 每个编码器和解码器块中，多头注意力（MHA）子层之后都紧跟着一个FFN子层。
# FFN引入非线性，MHA 模块本质上是由一系列矩阵乘法组成的，是线性操作
# FFN 通过其内部的 ReLU 或 GeLU 激活函数引入非线性
import torch
import torch.nn as nn
import torch.functional as F

class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        # 第一个线性层，输入维度dim，输出为hidden_dim
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        # 第二个线性层，输入维度为hidden_dim，输出为dim
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        # 定义dropout层，⽤于防⽌过拟合
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # 首先经过第一个线性层和RELU激活函数
        # 再经过第二个线性层和dropout
        return self.dropout(self.w2(F.relu(self.w1(x))))