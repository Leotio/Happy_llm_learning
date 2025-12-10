from transformers import PretrainedConfig

# 继承transformers库中的参数类PretrainedConfig
class ModelConfig(PretrainedConfig):
    # model_type作为配置文件和模型实现类之间的桥梁和身份ID
    '''AutoModel模型类的核心用途:
    定义网络结构：模型类在其 __init__ 方法中实例化了所有的层、模块、权重、偏置等
    定义前向传播逻辑forward: 包含了如何将输入数据通过网络结构进行计算，最终输出结果的逻辑
    管理权重：模型类是存储和管理网络参数的地方。加载预训练模型时，权重就是加载到这个类的实例中。
    model_type 存在的目的，是让自定义配置能够集成到 Hugging Face 生态系统中，实现配置与模型实现的自动映射
    比如后续保存了模型重新加载的时候,如果要用到AutoModel，没有这个model_type的话就没有办法加载出来，所以这个
    相当于是,便于Hugging Face生态下模型识别
    '''
    # AutoModel实例化并加载权重到正确的结构中:依赖 model_type 来找到对应的模型类，然后将权重加载到这个类实例中。
    model_type = "Tiny-K"
    def __init__(
            self,
            dim: int = 768, # 模型维度
            n_layers: int = 12,# Transformer层数
            n_heads: int = 16, # 注意⼒机制的头数
            n_kv_heads: int = 8, # 键值（注意力）头的数量
            vocab_size: int = 6144, # 词汇表⼤⼩
            hidden_dim: int = None, # 隐藏层维度
            multiple_of: int = 64, # 确保前馈网络的隐藏层维度是此值的倍数，以实现计算优化 
            # 现代 GPU 和 NPU执行矩阵乘法时，如果输入的张量维度是8、16、64的倍数，可以利用向量化指令和并行计算
            norm_eps: float = 1e-5, # 归⼀化层的eps
            max_seq_len: int = 512, # 最⼤序列⻓度
            dropout: float = 0.0, # dropout概率
            flash_attn: bool = True, # Flash Attention：内存优化的注意力机制实现,注意力机制的硬件加速
            **kwargs,# 允许用户或框架在不修改 ModelConfig 定义的情况下传递额外的参数。
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.flash_attn = flash_attn
        super().__init__(**kwargs)# 将捕获到的但未在当前子类中使用的参数，通过 **kwargs 的解包形式转发给父类的构造函数，确保父类能够完成其初始化工作
