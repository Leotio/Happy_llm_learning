'''初始化LLM'''

# import os
# # 设置环境变量，此处使⽤ HuggingFace 镜像⽹站
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# # 下载模型
# os.system('huggingface-cli download --resume-download Qwen/Qwen2.5-1.5B --local-dir E:\github\Happy_llm_learning\Chapert6')


"""
下面两段代码得到的都是没有经过预训练的模型，而⼀般情况下，很少从零初始化 LLM 进⾏预训练，
较多的做法是加载⼀个预训练好的 LLM 权重，在⾃⼰的语料上进⾏后训练
"""

'''加载定义好的模型参数,以 Qwen-2.5-1.5B 为例
使⽤ transforemrs 的 Config 类进⾏加载'''
# AutoConfig 类:通用的配置加载器
from transformers import AutoConfig
# 下载参数的本地路径
model_path = "E:\github\Happy_llm_learning\models"
# 从本地路径读取 config.json 文件,它只加载了模型的超参数
# config = AutoConfig.from_pretrained(model_path)


'''使⽤该配置⽣成⼀个定义好的模型,但它内部的权重是随机生成的数字,无法回答问题.通常用于从零开始训练'''
# 导入用于因果语言模型的类
# from transformers import AutoModelForCausalLM
# model = AutoModelForCausalLM.from_config(config,trust_remote_code=True)


"""
Transformers 框架中，模型和分词器被设计成两个独立的组件，因为它们处理的数据类型完全不同：
    模型是“大脑”，而 Tokenizer 是“感官（耳朵和嘴巴）”
    大脑虽然强大，但它无法直接处理空气中的声波（文字），它只能处理电信号（数字向量）
所以要让模型跑起来说话，应该使用下面两端代码得到 Model 和 Tokenizer
"""


'''可以直接得到一个有预训练权重的模型,用于推理'''
from transformers import AutoModelForCausalLM
# 直接从路径加载，不仅加载结构，还把阿里训练好的几GB权重填了进去
model = AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True)


# 加载⼀个预训练好的 tokenizer
from transformers import AutoTokenizer
# 加载 tokenizer.json 或 vocab.json 等文件
tokenizer = AutoTokenizer.from_pretrained(model_path)
