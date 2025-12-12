'''构建完整的LLaMA2模型: 即将Decoder Layer模块进行堆叠'''

import torch.nn as nn
import torch
import math
from typing import Optional
from s1_modelconfig import ModelConfig
from s2_rmsnorm import RMSNorm
from s3_group_query_attention import precompute_freqs_cis
from s5_Decoder_layer import DecoderLayer
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

# 继承链：Transformer -> PreTrainedModel -> nn.Module,
# 通过继承PreTrainedModel 这种方式，间接地、完整地继承了nn.Module 的所有功能，
# 同时还获得了 Hugging Face 提供的强大工具集，使得模型可以轻松地集成到大规模的 LLM 生态中。
class Transformer(PreTrainedModel):
    config_class = ModelConfig # 告诉PreTrainedModel基类应该使用哪个配置类来管理模型的超参数和结构信息
    # 该属性是可选的，它的值可以是torch.Tensor，也可以是 None(损失未计算时)
    last_loss: Optional[torch.Tensor] # 实例属性的类型提示 属性名称:类型注解

    def __init__(self, args: ModelConfig = None):
        # 调用父类PreTrainedModel的构造函数,而PreTrainedModel 需要接收配置对象args
        super().__init__(args)
        # 初始化模型参数
        self.args = args
        # 词汇表大小
        self.vocab_size = args.vocab_size
        # 层数
        self.n_layers = args.n_layers

        # 词嵌入层
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        # dropout层
        self.dropout = nn.Dropout(args.dropout)

        # Decoder层
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(DecoderLayer(layer_id, args))
        
        # 最终归一化层
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        # 输出层
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        
        # 将词嵌入层的权重与输出层的权重共享
        # 形状均为：N_vocab_size * D_model
        self.tok_embeddings.weight = self.output.weight

        # 预计算相对位置嵌入的频率
        freqs_cos, freqs_sin = precompute_freqs_cis(
            self.args.dim // self.args.n_heads, 
            self.args.max_seq_len)
        # persistent=False:该缓冲区不会被保存到模型的 state_dict 中
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # 初始化所有权重
        # self.apply(fn)是torch.nn.Module的一个方法,它的作用是：
        # 递归地遍历当前模块及其所有子模块,DecoderLayer,Attention,MLP,nn.Linear等，
        # 将指定的函数 fn 应用于每一个子模块
        self.apply(self._init_weights)

        # 残差的投影进行特殊的缩放初始化
        '''
        named_parameters(): PyTorch 模块的方法,递归地遍历当前模块及其所有子模块中所有可训练的参数
        pn (Parameter Name): 参数在模型中的完整路径名,例如 'layer.0.feed_forward.w3.weight'
        p (Parameter): 对应的参数张量本身
        '''
        for pn, p in self.named_parameters():
            # wo和w2是子层输出的最后一步，它们的输出直接加到残差连接上,通过缩放它们的权重，控制每次残差跳跃时引入的方差增量
            if pn.endswith('w2.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * args.n_layers))

        # 初始化最后一次前向传播的损失属性
        '''下面三行代码是Hugging Face PreTrainedModel 子类的 __init__ 构造函数末尾，
        以及 forward 函数开始之前常见的设置。它主要是为了兼容 Hugging Face 框架的输入/输出标准。'''
        self.last_loss = None
        # 准备一个标准的数据结构，用于封装模型在 forward 函数中的输出
        # 目的：在 Hugging Face的LLM 中，forward函数通常返回一个包含多个字段（如 loss, logits, past_key_values, hidden_states 等）的对象
        # CausalLMOutputWithPast是一种标准的输出格式，用于因果语言模型并支持键值缓存（KV Cache，即WithPast）
        self.OUT = CausalLMOutputWithPast()
        # 模块切分限制
        # 此处将所有模块的名称都放入这个列表，意味着禁止任何子模块被内部切分
        self._no_split_modules = [name for name, _ in self.named_modules]

    # 初始化权重
    def _init_weights(self, module):
        # 线性层权重初始化
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # 嵌入层权重初始化
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module, mean=0.0, std=0.02)
        
    
    def forward(self,tokens: torch.Tensor, targets: Optional[torch.Tensor] = None, **keyargs) -> torch.Tensor:
        '''
        -tokens: 通常是 Token ID 序列，例如 [1, 101, 5, 202, ...]
        -targets:目标输出：在训练时是标签（正确的下一个 Token ID），在推理时可能为 None
        -keyargs:捕获未显式定义的额外参数,兼容Hugging Face框架传递的各种参数,
                 如input_ids,attention_mask,past_key_values等
        '''
        # 两个if都是为了兼容Hugging Face库传递的参数命名
        if 'input_ids' in keyargs:
            # 如果用户或Trainer通过 input_ids关键字参数传入了Token ID，将本地变量tokens更新为这个标准输入
            tokens = keyargs['input_ids']
        if 'attention_mask' in keyargs:
            targets = keyargs['attention_mask']

        # 输入张量tokens的形状通常是 (Batch Size,Sequence Length)
        _bsz, seqlen = tokens.shape

        # 经过词嵌入层和dropout层
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)

        # 获取相对位置嵌入的频率(已经在init里面预计算出来了)
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        # 通过Deceoder层
        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)

        # 通过最终归一化层
        h = self.norm(h)

        # 当 targets 不为 None 时，模型处于 训练模式 或 评估模式，需要计算损失
        if targets is not None:
            # output：最后输出的线性层
            # logits为原始的对数几率，形状为(bsz, seqlen, vocab_size)
            logits = self.output(h)
            self.last_loss = F.cross_entropy(
                '''
                cross_entropyLogits要求输入的向量形状为:
                  Logits:(N, C)N是样本总数,C是类别数(词汇表大小)
                  Targets:(N),N是样本总数,包含每个样本的正确类别索引Token ID
                '''
                # 将logits从 (bsz, seqlen, vocab_size)展为(bsz * seqlen, vocab_size)
                logits.view(-1, logits.size(-1)), 
                targets.view(-1), # 目标标签张量（bsz,seqlen）展平为(bsz * seqlen)
                ignore_index=0, # 忽略标签值等于 0 的位置
                reduction='none', # 损失函数返回每个 Token 位置的单独损失值，而不是求平均
            )
        # 模型处于 推理模式 或 文本生成模式，不需要计算完整序列的损失。
        # 自回归文本生成中，只关心序列的最后一个 Token 的预测结果，因为这是下一个 Token 的概率分布。
        # 计算所有历史 Token 的 Logits 是浪费计算资源的
        else:
            # 只计算最后一个位置的输出进行前向传播
            # h：(bsz, seqlen, D_model)->取一个(bsz, 1, D_model)
            # logits则变为(bsz,1, vocab_size)
            logits = self.output(h[:, [-1], :])
            self.last_loss = None

        # 设置输出
        # __setitem__ 是 Python 字典或类似容器对象上的特殊方法
        # 等价于使用方括号进行赋值，即 self.OUT['logits'] = logit
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('last_loss', self.last_loss)
        return self.OUT


    @torch.inference_mode() # PyTorch装饰器，在生成过程中禁用梯度计算和跟踪
    # 文本生成函数的实现,使用自回归采样方法，在每次循环中生成一个新Token，直到达到最大长度或遇到停止Token
    def generate(self,
        idx, # 初始输入序列，形状为(bz, seq_len)。
        stop_id=None, # 可选参数，如果生成的 Token ID 等于此值，则停止生成
        max_new_tokens=256, # 模型将生成的新 Token 的最大数量，限制生成长度
        temperature=1.0, # 采样温度参数，用于控制生成结果的随机性（创造性）
        top_k=None, # Top-K 采样参数，限制采样时考虑的 Token 数量
        ):
        index = idx.shape[1]
        for _ in range(max_new_tokens):
            # 序列上下文过长，将他截断到最大长度,保留序列的最后max_seq_len个 Token
            idx_cond = idx if idx.size[1] < self.args.max_seq_len else idx[:, -self.args.max_seq_len:]

            # 前向传播获取序列中的最后一个位置的logits
            # self(idx_cond) 相当于 self.forward(idx_cond))，
            # 而且要用self(idx_cond) 才行，这样子才能去调用魔法方法，执行一些必要的操作
            # Logits 形状为 （bz,seqlen,vocab）
            logits = self(idx_cond).logits

            # 提取最后一个 Token 的 Logits 
            # PyTorch 中使用 整数索引（而不是切片 : 或 [a:b]）时，该维度会被自动压缩掉
            # 所以Logits为（bz,vocab）而不是（bz,1，vocab）
            logits = logits[:, -1, :]

            if temperature == 0.0:
                # torch.topk 返回 (值, 索引)，此处只取索引
                # 在最后一个维度（dim=-1）上找到logits最大（k=1）的那个Token
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            
            else:
                # 温度缩放：
                # 温度低 (接近0时)：分布变得“尖锐”，高分数的概率被放大，结果趋于确定性（类似贪婪）
                # 温度高 (>1)：分布变得“平坦”，低分数 Token 被选中的概率增加，结果更具随机性
                logits = logits / temperature
                if top_k is not None:
                    # 找出 Logits 最高的 k 个值
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    # v[:, [-1]]也就是最后一列，即第K大的logits值
                    # 将所有不在 Top-K 范围内的Logits值设置为极小的负数
                    # 确保后续的 Softmax 概率只在Top-K 的Token 上分配
                    logits[logits < v[:, [-1]]] = -float('Inf')

                #  Logits的值范围是(-inf, +inf)是模型直接输出的原始分数，要通过softmax才能变成概率分布
                # 将经过 Top-K 过滤的 Logits转化为概率分布
                probs = F.softmax(logits, dim=-1)

                # 根据概率分布 probs，随机抽取num_samples个样本
                # idx_next 是一个张量，包含下一个被选中的 Token 的 ID
                idx_next = torch.multinomial(probs, num_samples=1)
            
            if idx_next == stop_id:
                break

            # 更新序列，将新生成的Token ID(idx_next) 连接到当前序列idx的末尾
            # 以便在下一次循环中，模型可以将这个新 Token 作为上下文的一部分进行预测。
            idx = torch.cat((idx, idx_next), dim=1)

        # idx形状为 (Batch Size, Original Lengt+New Tokens)
        # 返回一个新的张量，它只包含 模型新生成的 Token 序列
        # 调用 generate 函数通常是为了得到模型“补全”或“回答”的部分
        return idx[:, index:]