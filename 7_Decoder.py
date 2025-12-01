'''Decoder Layer -> Decoder'''

import torch.nn as nn
'''Decoder Layer实现'''
# 由两个注意力层：1.mask_self_attention 2.multihead_attention和一个FFN构成
class DecoderLayer(nn.Module):
    def __init__(self. args):
        super().__init__()
    
    def forward(self, x):
        pass

'''Decoder的实现:N个Decoder Layer+Layer Norm'''
class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
    
    def forward(self, x):
        pass