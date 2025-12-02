'''Decoder Layer -> Decoder'''
'''
解码器的核心任是：
    理解自己： 通过掩码自注意力,理解目前已生成的（或已输入的）目标序列的内部上下文。
    理解源序列： 通过交叉注意力,将目标序列与编码器输出的源序列表示联系起来。
    交叉注意力的计算,query是Decoder的第一部分掩码自注意力的输出,而Key和Value采用Encoder的输出
'''

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

        # 第三部分为FFN,具体为MLP
        self.feed_forward = MLP(args)

    # enc_out在第二部分的多头注意力里面，充当Key和Value，建立目标序列与源序列之间的关系
    def forward(self, x, enc_out):
        # 先进行层归一化
        norm_x = self.attention_norm_1(x)

        # 掩码自注意力求解
        # happy llm给的代码是显式调用的forward方法
        # 但实际上，可以写成mask_attention(norm_x, norm_x, norm_x)让其自动调用，更好
        x = x + self.mask_attention.forward(norm_x, norm_x, norm_x)

        # 多头注意力
        norm_x = self.attention_norm_2(x)
        h = x + self.attention.forward(norm_x, enc_out, enc_out)

        # FFN
        norm_x = self.ffn_norm(h)
        out = h + self.feed_forward.forward(norm_x)

        return out


'''Decoder的实现:N个Decoder Layer+Layer Norm'''
class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        # N个DecoderLayer构成一个Decoder
        self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.n_layer)])
        self.norm = LayerNorm(args.embd)

    def forward(self, x, enc_out):
        for layer in self.layers:
            # nn.Module魔术方法会将此处传入的参数直接传给forward函数
            # 并调用forward函数进行计算更新
            x = layer(x, enc_out)
        return self.norm(x)