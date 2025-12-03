'''完整Transformer'''

import torch.nn as nn
import torch
import PositionalEncoding, Decoder, Encoder

class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        # 必须输入词表大小和block size
        # vocab_size=embedding层训练的权重矩阵的行，代表有多少个token的索引可以查找
        # 它定义的不是模型能够看到的所有词汇，而是它被训练来识别和生成的最大、最关键的词元集合,所以也是一个固定的值
        assert args.vocab_size is not None
        # block_size：模型单次能处理的词元数量 
        # 他是embedding层训练的权重矩阵的列，注意block size是单词可以处理的Token的数量，也经常取512
        # 但是并不是d_model，d_model是对于单个token的表示的维度，只不过也经常会出现512的设定
        assert args.block_size is not None

        self.args = args
        
        # 这里用ModuleDict的话比ModuleList更方便通key来访问子模块
        # 把所有核心组件在此处定义
        self.transformer = nn.ModuleDict(dict(
            # word token embedding：首先进行词嵌入
            wte = nn.Embedding(args.vocab_size, args.n_embd),
            # word position encoding：进行位置编码
            wpe = PositionalEncoding(args),
            # dropout层
            drop = nn.Dropout(args.dropout),
            # encoder编码器
            encoder = Encoder(args),
            # decoder解码器 
            decoder = Decoder(args)
        ))

        # 最后的线性层，输入为n_embd,输出为vocab_size，代表的就是输入了多少

        # 初始化所有的权重
        self.apply(self.__init__weights)

        # 查看一下所有参数的数量
        # 1e6 = 1 × 10^6
        print("number of parameters: %.2fM" % (self.get_num_param()/1e6))