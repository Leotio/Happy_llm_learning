'''Layer Norm层实现'''
import torch
import torch.nn as nn 
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6) :
        super().__init__()
        # nn.Parameter(...): PyTorch 的包装器，将普通的Tensor注册为可学习的参数
        # 缩放因子: 初始化为全 1 的向量,允许归一化后的数据被重新缩放
        self.a_2 = nn.Parameter(torch.ones(features))
        # 偏移因子: 初始化为全 0 的向量,允许归一化后的数据被重新偏移
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        # x:(B, L, d_model)->mean:(B,L,1)
        mean = x.mean(-1, keepdim=True) 
        # std:(B,L,1)
        std = x.std(-1, keepdim=True) 
        # mean,std的维度是(B, L, 1)，x是(B, L, d_model)，广播机制会自动将mean和std复制d_model次
        # a_2和b_2将归一化后的结果进行缩放和平移，确保模型在归一化后仍能学习复杂的分布
        return self.a_2 * (x - mean) /(std + self.eps) + self.b_2
        # 最终输出的维度与x的维度相同