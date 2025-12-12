'''构建完整的LLaMA2模型: 即将Decoder Layer模块进行堆叠'''

import torch.nn as nn
import torch
from typing import Optional
from s1_modelconfig import ModelConfig
from s2_rmsnorm import RMSNorm
from s3_group_query_attention import precompute_freqs_cis
from s5_Decoder_layer import DecoderLayer
import torch.nn.functional as F
from transformers import PreTrainedModel

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
        self.layer = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layer.append(DecoderLayer(layer_id, args))
        
        # 归一化层
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        # 输出层
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        
        # 将词嵌入层的权重与输出层的权重共享
        self.tok_embeddings.weight = self.output.weight

        # 预计算相对位置嵌入的频率
        


        # 初始化所有权重

        # 残差的投影进行特殊的缩放初始化

        # 初始化最后一次前向传播的损失属性

    # 初始化权重
    def _init_weights(self, module):
        return super()._init_weights(module)
    
    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None, **keyargs) -> torch.Tensor:
        '''
        -tokens: 
        -targets:
        -keyargs:
        '''
    
    @torch.inference_mode()
    def generate(self, idx, stop_id=None, max_new_tokens=256, temperature=1.0, top_k=None):
       '''

       '''