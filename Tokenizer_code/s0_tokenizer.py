'''使用Hugging Face的Tokenizers库来训练一个BPE Tokenizer'''
import random
import json
import os
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import (
decoders,
models,
pre_tokenizers,
trainers,
Tokenizer,
)
from tokenizers.normalizers import NFKC
from typing import Generator

'''加载训练数据'''

'''创建配置文件'''

'''训练BPE Tokenizer'''

'''使用训练好的Tokenizer'''