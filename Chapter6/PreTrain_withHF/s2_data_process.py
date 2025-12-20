'''处理用于预训练的数据'''

# 1------加载预训练数据------

from datasets import load_dataset
from s1_init_llm import tokenizer
from itertools import chain

ds = load_dataset('json', data_files='/mobvoi_seq_monkey_general_open_corpus.jsonl')

# 查看特征
column_names=list(ds["train"].features)

# 2------对数据进行tokenizer------

# 使用加载好的tokenizer对数据集进行处理
# Hugging Face 的 datasets 在处理数据时，为了效率通常按照Batch处理
# examples 的结构类似于：{"text": ["句子1", "句子2", "句子3"], "label": [0, 1, 0]}
# examples["text"] 拿到的就是一个包含多个字符串的列表
def tokenize_function(examples):
    # tokenizer进行：
    # 1.分词：把句子切成词或子词（如 "你好" -> "你", "好"）
    # 2.映射：根据词表把词换成数字（如 "你" -> 108）
    # 3.增加特殊标记：在开头加上 [CLS] (开始符号) 或结尾加上 [SEP] (分隔符号)

    # output 返回一个字典，通常包含以下关键信息：
    # input_ids数字索引列表告诉模型具体是哪些词
    # attention_mask 0 和 1 组成的序列告诉模型哪些是真正的词 (1)，哪些是补齐的填充位 (0)
    output = tokenizer([item for item in examples["text"]])
    return output

# 批量处理
tokenized_datasets = ds.map(
    tokenize_function, # 告诉map方法,对数据集里的每一行数据，具体执行什么转换操作
    batched=True,
    num_proc=10, # 10 个 CPU 进程并行处理
    # 移除原始列,原始数据里包含 text, source, metadata 等文本列,
    # 分词后，得到input_ids 和 attention_mask，原始的列都没有用了
    remove_columns=column_names, 
    load_from_cache_file=True, # 把结果保存在缓存文件夹里（.cache/huggingface/datasets）
    desc="Running tokenizer on dataset", # 设置进度条说明
)

# 3------将文本拼接到统一长度的文本块------

# 在此处将块长取为2048
block_size = 2048

def group_texts(examples):
    # 将⽂本段拼接起来
    # examples.keys()：包含 input_ids 和 attention_mask
    # chain(*examples[k])：假设 examples["input_ids"]是 [[1, 2], [3, 4, 5]]，chain 会把变成一个单一的长序列：[1, 2, 3, 4, 5]
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    # 计算拼起来的整体⻓度
    # list(examples.keys()):将examples的key转为列表->然后[0]返回"input_ids"
    # concatenated_examples["input_ids"]：通过键名拿到那个已经拍平的超长列表
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    # 如果⻓度太⻓，进⾏分块
    if total_length >= block_size:
        # 取block_size整数倍长度截断
        total_length = (total_length // block_size) * block_size
    result = {
        # 把k对应的超长列表，切分成等长小列表
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        # items()返回的是键值对元组，k会先后变为"input_ids"和attention_mask
        for k, t in concatenated_examples.items()
    }

    # CLM 任务，labels 和 input 是相同的
    # 在CLM训练中，CLM 任务的目标是预测下一个词,为了让模型知道它预测得对不对，我们需要给它一份“标准答案”（Labels）
    # 而在预测下一个词的任务中，标准答案其实就是文本本身
    result["labels"] = result["input_ids"].copy() # .copy() 确保 labels 是一份独立的副本
    return result

# 批量处理
lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    num_proc=10,
    load_from_cache_file=True,
    desc=f"Grouping texts in chunks of {block_size}",
    # datasets库会从磁盘一次性读取40000条记录，并将它们打包成一个字典传给 group_texts
    batch_size = 40000,
)

# lm_datasets 通常是一个 DatasetDict 对象,里面装有 train、test和 validation
# train_dataset 就是⼀个可直接⽤于 CLM Pretrain 的预训练数据集了，每个样本⻓度为2048个token
train_dataset = lm_datasets["train"]