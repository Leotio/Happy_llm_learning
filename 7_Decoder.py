'''Decoder Layer -> Decoder'''

import torch.nn as nn
'''Decoder Layer实现'''
# 由两个注意力层：1.mask_self_attention 2.multihead_attention和一个FFN构成
class DecoderLayer(nn.Module):
    def __init__(self. args):
        super().__init__()
        # 每层有三个LayerNorm，一个在掩码注意力分数之前，一个在多头注意力之前，一个在FFN之前
        self.attention_norm_1 = LayerNorm(args.n_embd)

        # 第一部分为掩码自注意力，is_casual设为True
        self.mask_attention = MultiHeadAttention(args, is_causal=True)
        self.attention_norm_2 = LayerNorm(args.n_embd)

        # 第二部分为多头注意力，is_casual设为False
        self.attention = MultiHeadAttention(args, is_causal=False)
        self.ffn_norm = LayerNorm(args.n_embd)

        # 第三部分为FFN
        self.feed_forward = MLP(args)

    # enc_out在第二部分的多头注意力里面，充当Key和Value，建立目标序列与源序列之间的关系
    def forward(self, x, enc_out):
        # 先进行层归一化


'''Decoder的实现:N个Decoder Layer+Layer Norm'''
class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
    
    def forward(self, x):
        pass