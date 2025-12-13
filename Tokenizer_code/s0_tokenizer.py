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

'''加载训练数据,读取JSONL⽂件并安全提取⽂本数据'''
def read_texts_from_jsonl(file_path: str) -> Generator[str, None, None]:
    with open(file_path, 'r', encoding='utf-8') as f:
        # enumerate(iterable, start=0)：start指定索引开始的值
        for line_num, line in enumerate(f, 1):
            try:
                # 使用json.loads() 函数将lines反序列化为一个字典对象data
                data = json.loads(line)
                # 确保该行包含'text'字段
                if 'text' not in data:
                    raise KeyError(f"Missing 'text' field in line {line_num}")

                # yield：将 'text' 字段的值作为一个结果输出，函数的状态，包括所有局部变量被保存
                yield data['text']
            
            # line不是一个合法的JSON格式字符串，如缺少必要的引号、多余的逗号、JSON 格式不规范等
            except json.JSONDecodeError:
                print(f"Error decoding JSON in line {line_num}")
                continue
            # JSON解析成功，但解析后的字典data中缺少程序期望的关键字段'text'
            except KeyError as e:
                print(e)
                continue



'''创建配置文件'''

'''训练BPE Tokenizer'''

'''使用训练好的Tokenizer'''