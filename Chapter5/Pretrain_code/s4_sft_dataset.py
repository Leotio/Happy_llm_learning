'''
SFT Dataset是⼀个多轮对话数据集,我们的⽬标是让模型学会如何进⾏多轮对话。
在这个阶段我们的输⼊是上⼀轮的对话内容，输出是当前轮的对话内容
'''

import json
from torch.utils.data import Dataset

class SFTDataset(Dataset):
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
    
    # 为多轮对话模型的训练数据生成损失掩码 (loss_mask),只在模型应该生成回复的位置计算损失，忽略用户的指令
    def generate_loss_mask(self, input_ids):
        mask = [0] * len(input_ids)
        # a_sequence是Assistant回复的起始标志， Token ID 序列代表字符串：<|im\_start|>assistant\n
        a_sequence = [3, 1074, 537, 500, 203]
        a_length = len(a_sequence)
        n = len(input_ids)
        i = 0

        # 确定 i 开始的连续 a_length 个 Token ID，是否与预定义 a_sequence 完全一致
        while i <= n-a_length:
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
        sample = json.loads(self.data[index])

        # 将一个包含角色（如Assistant）和内容的对话列表，转换为单一的、符合模型训练要求的文本字符串
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