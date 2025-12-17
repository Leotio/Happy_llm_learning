'''
加载已经预处理好的数据集,将文本数据转化为模型能够理解的Token
准备模型训练的输入X、下一个词的预测Y,以及告诉模型哪些部分需要计算损失的掩码 (loss_mask)
'''
import json
import torch
import numpy as np
from torch.utils.data import Dataset

# 继承自PyTorch的抽象基类Dataset,创建自定义数据集
class PretarinDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = 0
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = f.readlines()
    
    # PyTorch要求 Dataset类必须实现这个方法，以返回数据集中的样本总数
    def __len__(self):
        # 返回之前加载的行数，即数据集的样本数量。
        return len(self.data)
    
    # 负责处理和转换单个数据样本
    def __getitem__(self, index: int):
        sample = json.loads(self.data[index])
        
        # 从字典中提取 'text'，并在其前面添加一个BOS标记（例如：[CLS] 或 <s>）
        # 这是预训练语言模型（LLM）的常见做法
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
        # 因为 Y 比 input_id 少一个元素，loss_mask 必须与 Y 保持长度一致
        # 只看 Y 中那些对应的 loss_mask 值为 1 的位置，而忽略（不计算损失）loss_mask 值为 0 的位置
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)

        # NumPy 数组转换为 PyTorch 张量 (torch.from_numpy)，并以 (X, Y, loss_mask) 的元组形式返回
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)