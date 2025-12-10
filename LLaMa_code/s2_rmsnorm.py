'''实现RMSNorm'''
import torch.nn as nn
import torch
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        # weight为可学习参数，全部初始化为1
        # 即最后的缩放因子参数
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # RMSNorm就是一个LayerNorm的简化版本，归一的时候不在x上减去均值μ了
        # x.pow(2).mean(-1, keepdim=True)得到x方的均值 + eps
        # 如果没有keepdim=True会导致和x相乘失败！
        # torch.rsqrt 对（）中的表达式开根并取倒数
        # x乘上这个倒数就得到了RMSNorm中除了放缩参数外的表达式
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        # 归一化操作中精度要求高，转为float计算再转回去
        output = self._norm(x.float()).type_as(x)
        # weight是RMSNorm中一个可学习的缩放因子参数
        return output * self.weight
