'''Encoder Layer -> Encoder'''
from s4_layer_norm import LayerNorm
from s3_ffn import MLP
from s2_multi_head_attention import MultiHeadAttention

import torch.nn as nn
'''Encoder Layer实现'''
# 由一个注意力层和一个FFN构成
class EncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        # 每层有两个LayerNorm，一个在计算注意力分数之前，一个在进入FFN之前
        # args.n_embd是嵌入模型的维度，也就是特征数
        self.attention_norm = LayerNorm(args.n_embd)
        # Encoder不需要掩码，is_causal=False
        self.attention = MultiHeadAttention(args, is_causal=False)
        self.ffn_norm = LayerNorm(args.n_embd)
        self.feed_forward = MLP(args)
    
    def forward(self, x):
        # 计算atention前先进行一次曾归一化
        norm_x = self.attention_norm(x)
        # 计算注意力分数,encoder计算自注意力
        h = x + self.attention.forward(norm_x, norm_x, norm_x)
        # 再次进行层归一化后，进入FFN
        out  = h + self.feed_forward.forward(self.ffn_norm(h))
        return out
    
'''Encoder的实现:N个Encoder Layer+Layer Norm'''
class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        # 一个Encoder由n_layer个EncoderLayer构成
        # nn.ModuleList:将一个包含多个 nn.Module 实例的列表，作为一个整体注册到父模块中
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.n_layer)])
        self.norm = LayerNorm(args.embd)

    def forward(self, x):
        '''分别通过 N 层EncoderLayer'''
        for layer in self.layers:
            # 执行x = layer(x)，PyTorch 实际执行的步骤是：
            # 调用 __call__ 方法：首先调用 nn.Module基类中定义的魔术方法 __call__(self, *input, **kwargs)
            # 执行 Hooks：__call__ 会处理所有的前置和后置 Hook（例如，用于调试或可视化）。
            # 调用 forward： __call__ 方法的核心任务是调用在EncoderLayer中定义的 forward(self, x) 方法
            # 调用 __call__ 间接调用 forward是 PyTorch 自动化管理模块状态（如模式切换、梯度计算、Hooks）的基础
            x = layer(x)
        # 最后再进行一次归一化
        return self.norm(x)