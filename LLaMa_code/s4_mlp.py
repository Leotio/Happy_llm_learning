'''LLaMA的MLP模块的实现:SwiGLU(或简称 GLU 门控)前馈网络'''

import torch.nn as nn
import torch.nn.functional as F
from s1_modelconfig import ModelConfig

class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        # 如果没有指定隐藏层的维度，则将其设定为输入维度的4倍
        # 然后将其减少到2/3，确保他是multiple_of的倍数
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        
        # 定义第一层线性变换，从输入维度到隐藏维度
        # 门控分支：用于激活函数 SiLU 的输入
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)

        # 定义第二层线性变换，从隐藏维度到输入维度
        # 输出投影：将特征从隐藏维度投影回模型维度。
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

        # 定义第三层线性变化，从输入维度到隐藏维度
        # 主分支：用于提供门控的未激活输入
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

        # 定义dropout层，避免过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        ''' 
        x首先经过第一层线性变化,然后通过SILU激活函数
        x经过第三层线性变换的结果和F.silu(self.w1(x))相乘
        最后经过第二层线性变化和dropout
        '''
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
   
    # SiLU 激活后的门控信号（值域在 0 到 1 之间）对主信息分支进行 “门控”
    # 如果门控信号接近 1，则信息完全通过；
    # 如果接近 0，则信息被阻塞。这允许模型动态地控制哪些特征应该被保留和增强。
        
