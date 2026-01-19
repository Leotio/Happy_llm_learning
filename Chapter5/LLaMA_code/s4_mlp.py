'''LLaMA的MLP模块的实现:SwiGLU(或简称 GLU 门控)前馈网络'''

import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        # 传统的 MLP 只有两层矩阵，维度通常翻 4 倍。
        # SwiGLU 有三层矩阵（w1, w2, w3），为了让总参数量和计算量与传统 MLP 保持一致，需要将维度缩小到原来的 2/3。
        
        # 如果没有指定隐藏层的维度，则将其设定为输入维度的4倍
        # 然后将其减少到2/3，确保他是multiple_of的倍数(将维度强制设为 8 或 256 的倍数，可以大幅提高矩阵乘法的并行效率)
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # w1	门控通道	负责生成“开关”信号，决定哪些信息重要。
        # w3	内容通道	负责提取原始的、未加工的语义特征。
        # w2	结果投影    将加工后的高维信息压缩回原始维度，方便后续层处理。
        
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
        silu(self.w1(x))相当于生成一个开关，决定什么重要
        self.w3(x) 就是内容，只不过是升了下维度
        开关和内容相乘得到结果，再通过用于降维度的w2就可以得到最后的输出了
        当然还要一次dropout
        '''

        # 注意*运算是按元素相乘！！
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
   
    # SiLU 激活后的门控信号（值域在 0 到 1 之间）对主信息分支进行 “门控”
    # 如果门控信号接近 1，则信息完全通过；
    # 如果接近 0，则信息被阻塞。这允许模型动态地控制哪些特征应该被保留和增强。
        
