'''此代码完成数据预处理，准备用于Pretrain和SFT的数据，将文本数据转化为模型能够理解的Token'''


import json
import torch
import numpy as np
from torch.utils.data import Dataset
import random
import re
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os

'''
用于Pretrain的Dataset一般是纯文本、文章、代码，而不是“对话格式”。因为Pretrain的⽬标是让模型学会预测下一个词
'''
# 继承自PyTorch的抽象基类Dataset,创建自定义数据集
class PretarinDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = 0
        
        # 预计算每行的起始字节偏移量
        self._offsets = []
        with open(data_path, 'rb') as f:
            self._offsets.append(0) 
            while f.readline():
                # f.tell() 返回当前文件指针在文件中的物理位置（第几个字节）
                self._offsets.append(f.tell())

        # 最后一次 tell() 到达了文件末尾EOF，减去 1 才是真正的有效行数
        self._total_lines = len(self._offsets) - 1 

    # PyTorch要求 Dataset类必须实现这个方法，以返回数据集中的样本总数
    def __len__(self):
        # 返回之前加载的行数，即数据集的样本数量。
        return self._total_lines
    
    # 负责处理和转换单个数据样本
    def __getitem__(self, index: int):
        with open(self.data_path, 'rb') as f:
            # 利用之前算好的偏移量seek，直接跳到硬盘的某个位置
            f.seek(self._offsets[index])
            # 读出那一行
            line = f.readline().decode('utf-8')
        sample = json.loads(line)
        
        # 在text前面加一个BOS
        text = f"{self.tokenizer.bos_token}{sample['text']}"

        # 用tokenizer将text转换为ID序列，然后提取生成的 'input_ids' 列表，并将序列截断到最大长度
        input_id = self.tokenizer(text).data['input_ids'][:self.max_length]
        # 记录截断后的实际文本 ID 长度
        text_len = len(input_id)

        # 将截断后的文本补到max_length
        padding_len = self.max_length - text_len
        input_id = input_id + [self.padding] * padding_len

        # 创建一个掩码列表：实际文本 ID 对应 1（需要计算损失），填充 ID 对应 0（不计算损失）
        loss_mask = [1] * text_len + [0] * padding_len

        input_id = np.array(input_id)
        # 创建模型的输入张量 X,包含原始 input_id 中除了最后一个 ID 的所有 ID
        X = np.array(input_id[:-1]).astype(np.int64)
        # 目标张量 Y,包含原始 input_id 中除了第一个ID的所有ID,Y相当于对 X 中每个词的预测目标（即下一个词）
        Y = np.array(input_id[1:]).astype(np.int64)

        # 将 loss_mask 转换为 NumPy 数组，并截断第一个元素
        # Y是需要模型预测的，所以loss_mask要和Y对齐才行
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)

        # NumPy 数组转换为 PyTorch 张量 (torch.from_numpy)，并以 (X, Y, loss_mask) 的元组形式返回
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)

'''
用于SFT的Dataset是⼀个多轮对话数据集,因为我们的⽬标是让模型学会如何进⾏多轮对话，在SFT阶段就需要完成这件事
SFT的 输⼊是上⼀轮的对话内容，输出是当前轮的对话内容
'''
class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = 0
        with open(data_path, 'rb') as f:
            self._offsets.append(0)
            while f.readline():
                self._offsets.append(f.tell())
        self._total_lines = len(self._offsets) - 1 
    
    # PyTorch要求 Dataset类必须实现这个方法，以返回数据集中的样本总数
    def __len__(self):
        # 返回之前加载的行数，即数据集的样本数量
        return self._total_lines
    
    # 为多轮对话模型的训练数据生成损失掩码 (loss_mask),只在模型应该生成回复的位置计算损失，忽略用户的指令
    def generate_loss_mask(self, input_ids):
        mask = [0] * len(input_ids)

        # a_sequence去获取Assistant回复的起始标志 <|im\_start|>assistant\n 对应的Token ID 序列
        a_sequence = self.tokenizer("<|im_start|>assistant\n")['input_ids'] 
        a_length = len(a_sequence)

        # 完整输入的id的长度
        n = len(input_ids)
        

        # 从 i 开始的连续 a_length 个 Token ID，是否与预定义 a_sequence 完全一致
        # 要完全一致才是正确的assistance的回答开始
        i = 0
        while i <= n-a_length: # 用while循环！！！支持多轮对话！！！！
            match = True
            for k in range(a_length):
                if input_ids[i+k] != a_sequence[k]:
                    match = False
                    break
            # 如果找到了一个Assistant回复的起始标志
            if match:
                # 就开始查找第⼀个4, 4 为 <|im_end|> EOS id
                j = None
                for idx in range(i + a_length, n):
                    if input_ids[idx] == 4:
                        j = idx
                        break
                
                # j不为None，就说明找到了结束标志
                if j is not None:
                    start = i + a_length
                    end = j
                    if start <= end:
                        # 找到了助手回复的起始标记 (a_sequence) 和结束标记 (<|im_end|>，即 ID 4)，
                        # 就将它们之间的所有 Token 对应的 mask 值设置为 1，从而激活这些 Token 的损失计算
                        for pos in range(start, end+1):
                            if pos < len(mask):
                                mask[pos] = 1
                # 更新 i 值，继续while循环        
                i += a_length
            else:
                i += 1
        return mask
    

    def __getitem__(self, index: int):
        with open(self.data_path, 'rb') as f:
            f.seek(self._offsets[index])
            line = f.readline().decode('utf-8')
        sample = json.loads(line)

        # 将一个包含角色和内容的对话列表，按照chat_template转换为单一的、符合模型训练要求的文本字符串
        # 比如 <|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n你好！我是AI。<|im_end|>\n 
        # tokenize=False 表示只返回字符串，先不转 ID
        text = self.tokenizer.apply_chat_template(sample, tokenize=False, add_generation_prompt=False)

        # 用tokenizer将text转换为ID序列，然后提取生成的 'input_ids' 列表，并将序列截断到最大长度
        input_id = self.tokenizer(text).data['input_ids'][:self.max_length]
        # 记录截断后的实际文本 ID 长度
        text_len = len(input_id)

        # 将截断后的文本补到max_length
        padding_len = self.max_length - text_len
        input_id = input_id + [self.padding] * padding_len

        # 生成损失掩码
        loss_mask = self.generate_loss_mask(input_id)


        # SFT 没有改变这个原理，还是预测下一个词，只是mask不太一样，只需对 Assistant 的话负责，user的话都不算loss
        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)