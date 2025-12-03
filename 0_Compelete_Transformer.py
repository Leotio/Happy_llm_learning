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

        # 最后的线性层，输入为n_embd,输出为vocab_size维，代表的就是词表中各个token的选取概率
        # 语言模型头部Language Model Head:模型执行最终预测，将隐藏状态转换为词元概率的关键步骤
        self.lm_head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        # 初始化所有的权重
        # 确保了无论模型结构多么复杂，其中的每一个可训练参数都能在模型启动训练前得到统一、规范的初始化
        '''
            apply():nn.Module方法,PyTorch模块都拥有的一个内置方法。
            它接受一个函数作为输入，并对当前模块及其所有子模块应用该函数。
            self.__init__weights:初始化函数,当前类中定义的私有方法
            apply() 方法会将模型中的每个子模块作为参数传递给这个函数
        '''
        self.apply(self.__init__weights)

        # 查看一下所有参数的数量
        # 1e6 = 1 × 10^6
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    '''统计一下参数的总数量'''
    # non_embedding:是否统计embedding的参数
    # 在评估模型的大小时，排除嵌入参数是很有用的，因为嵌入层通常只包含一个大的查找表，
    # 而模型的真正计算和逻辑能力主要体现在 Transformer 块（非嵌入部分）
    def get_num_params(self, non_embedding=False):
        # parameters():nn.Module 的内置方法，返回一个迭代器，包含模型（self）中所有可训练的参数（权重和偏置）张量。
        # p.numel(): 对于迭代器返回的每个p，numel()方法返回该张量中元素的总数
        # 即该参数占用的内存单元数量，这里的 “内存单元数量” 是对 numel() 结果的一个更物理层面或计算机科学层面的描述，因为每个元素（例如一个 32 位浮点数）确实占据了内存中的一个单元。）
        n_params = sum(p.numel() for p in self.parameters())

        if non_embedding:
            # self.transformer.wte: 访问模型中存储词元嵌入层的模块
            #.weight: 访问嵌入层 nn.Embedding 模块中实际存储词元嵌入的权重张量
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    '''初始化权重'''
    def _init_weights(self, module):
        # 对于线性层,权重矩阵使用正态分布进行初始化,旨在保持激活值的方差稳定。
        # 同时，如果线性层包含偏置项，则将其初始化为 零。
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                # 下划线 _ 表示这是原地操作，直接修改张量
                torch.nn.init.zeros_(module.bias)
        
        # 对于嵌入层，词元嵌入矩阵同样使用正态分布进行初始化
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    '''前向计算函数'''
    # idx为输入(batch size,sequence len)如果达不到sequence len的序列，会进行padding
    # target 为目标输出，用于计算loss
    def forward(self, idx, targets=None):
        # PyTorch张量对象的核心属性之一 .device，它存储了该张量当前被分配到的计算设备
        device = idx.device
        # 获取b，t
        b, t = idx.size()
        # assert condition, error_message
        assert t <= self.args.block_size, f"不能计算该序列，该序列长度为{t},最大序列长度只有{self.args.block_size}"

        print("idx",idx.size())
        # idx首先通过embedding层->（batch_size, sequence_len,n_embd）
        # nn.ModuleDict 继承自 nn.Module，它允许通过属性风格（点号）来访问其内部注册的子模块，如此处的.wte
        tok_emd = self.transformer.wte(idx)
        print("tok_emd",tok_emd.size())

        # 通过位置编码层
        pos_emd= self.transformer.wpe(tok_emd)
        print("pos_emd",pos_emd.size())

        # 进行一次dropout
        x = self.transformer.drop(pos_emd)
        print("x after wpe&dropout:",x.size())

        # 通过Encoder
        enc_out = self.transformer.encoder(x)
        print("enc_out",enc_out.size())

        # 通过Decoder
        x = self.transformer.decoder(x, enc_out)
        print("x after decoder:",x.size())

        # 此时x为(batch_size, sequence_len,d_model)

        # 训练阶段
        if targets is not None:
            # 训练需要计算loss
            # 首先通过最后的线性层->(batch_size,sequence_len,vocab_size)
            logits = self.lm_head(x)
            # 再和targets计算交叉熵
            '''
            logits.view(-1, logits.size(-1)):
                logits 的原始形状: (B, T, V),logits.size(-1): 获取 Logits 的最后一个维度的大小词汇表大小
                .view(-1, V) 将张量重塑为一个二维矩阵
                第一个参数-1自动计算所需的第一维度大小;第二个参数确保了第二维度是词汇表大小
                每一行是模型对下一个词元的V种可能性的预测分数。
            targets.view(-1):
                targets 的原始形状: (B, T),包含每个位置的真实词元 ID。
                .view(-1): 将目标序列展平为一个一维向量(B×T)
                每一项是对应位置的真实词元 ID
            ignore_index=-1:
                在计算损失的平均值时，忽略目标标签中值等于 -1 的所有位置。
                作用： 如果序列中的某个词元(Padding 词元）被分配了 ID -1
                那么模型对该位置的预测误差将不会计入最终的损失值中,确保了填充部分不会影响模型的训练。
            '''
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        
        # 推理阶段，只需要logits，无需计算loss
        else:
            # x为(batch_size, sequence_len,d_model)
            # 在执行预测推理过程的时候，我们没有必要去把序列的所有token都预测，
            # 只需要每次预测一个，每次预测最后一个作为输出，可以节约资源
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        
        return logits, loss