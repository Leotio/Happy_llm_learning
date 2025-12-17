'''LLaMA的DecoderLayer实现 : 其实就是将GQA模块和MLP模块进行一个组合'''

import torch.nn as nn
from s1_modelconfig import ModelConfig
from s3_group_query_attention import Attention
from s4_mlp import MLP
from s2_rmsnorm import RMSNorm

class DecoderLayer(nn.Module):
    def __init__(self, layer_id: int, args: ModelConfig):
        super().__init__()
        # 定义多头注意力的头数
        self.n_heads = args.n_heads
        # 定义输入维度
        self.dim = args.dim
        # 定义每个头的维度，等于输入维度除以头数
        self.head_dim = args.dim // args.n_heads

        # 定义LLaMA2 Attention对象，用于进行多头注意力计算
        self.attention = Attention(args)

        # 定义LLaMA2 MLP对象，用于进行前馈神经网络计算
        self.feed_forward = MLP(
            dim = args.dim,
            hidden_dim = args.hidden_dim,
            multiple_of = args.multiple_of,
            dropout = args.dropout,
        )

        # 定义层的ID
        # layer_id 相当于这个神经网络层在整个模型中的身份证号码，用于所有需要识别、区分和配置不同层行为的场景
        self.layer_id = layer_id

        # 定义注意力计算的归一化层
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # 定义前馈神经网络的归一化层
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin):
        # LLaMA都是pre norm，先归一化操作再进行其他操作
        '''
        self.attention:GQA计算对象, 调用他的forward函数并传入需要的参数
        x原始输入+注意力计算结果 传入MLP中
        '''
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out

