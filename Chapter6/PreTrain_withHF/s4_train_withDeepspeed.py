'''使用Deepsped框架实现分布式训练'''

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from torchdata.datapipes.iter import IterableWrapper
from itertools import chain
import deepspeed
from typing import Optional,List
import datasets
import pandas as pd
import torch
from datasets import load_dataset
import transformers
from transformers import (
AutoConfig,
AutoModelForCausalLM,
AutoTokenizer,
HfArgumentParser,
Trainer,
TrainingArguments,
default_data_collator,
set_seed,
)
import datetime
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
import swanlab

logger = logging.getLogger(__name__)

""" 1------定义⼏个超参的类型，⽤于处理 sh 脚本中设定的超参值。"""

# @dataclass装饰器告诉 Python，这个类主要是用来存储数据的，自动生成 __init__方法
@dataclass
class ModelArguments:
    '''
    模型相关的超参
    '''
    # 采用Python Dataclass（数据类）结合 Hugging Face field 元数据的写法
    # 让 transformers 的 HfArgumentParser 能够自动解析命令行参数

    # 变量名: 类型 = 默认值
    # field 是标准库 dataclasses 里的函数，用来给这个变量增加更多“配置”
    model_name_or_path: Optional[str] = field(
        default=None, # 默认为 None
        metadata={
            # 运行 python train.py --help 时，屏幕上会显示这个help中的参数说明
            "help": (
                "后训练使⽤，为预训练模型参数地址"
            )
        },
    )

    config_name: Optional[str] = field(
        default=None, metadata={"help": "预训练使⽤，Config ⽂件地址"}
    )

    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "预训练 Tokenizer 地址"}
    )

    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "模型训练使⽤的数据类型，推荐 bfloat16"
            ),
            # 白名单限制，只能从这几个选项里选，减少因拼写错误导致的训练失败
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )


@dataclass
class DataTrainingArguments:
    '''
    数据相关的超参
    '''
    train_files: Optional[List[str]] = field(default=None, metadata={"help": "训练数据路径"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "设置的⽂本块⻓度"
            )
        },
    )

    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "预处理使⽤线程数."},
    )


def main():
    # 加载脚本参数
    # 实例化解析器
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    # 执行解析与对象实例化
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 初始化 SwanLab，在本地脚本和 SwanLab 可视化平台之间建立一个通信通道
    # experiment_name：自己定义一个实验的名字
    swanlab.init(project="pretrain", experiment_name="from_scrach")

    """ 2------使用logging库实现日志记录"""

    # 日志设置 
    logging.basicConfig(
        '''format定义每一行日志长什么样:
            asctime: 发生时间; levelname:信息级别INFO/WARNING/ERROR
            name: 哪个模块产生的日志; message:具体的文字内容'''
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        # 指定时间的具体格式（月/日/年 时:分:秒）
        datefmt="%m/%d/%Y %H:%M:%S",
        # 指定日志去哪里,这里是发往 stdout（标准输出），即控制台屏幕
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # 将⽇志级别设置为 INFO,低于这个级别（如 DEBUG）的信息将被隐藏，只显示关键的初始化信息
    transformers.utils.logging.set_verbosity_info()
    # 之前解析出来的 training_args 中获取用户设定的日志级别
    log_level = training_args.get_process_log_level()

    # 确保 logger、Hugging Face的 datasets 库和 transformers 库步调一致
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)

    # 标准化输出
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # 训练整体情况记录
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    """ 3------间隔保存checkpoint"""

    # 预设一个空变量，用来存放找到的最新存档路径
    last_checkpoint = None
    # 检查sh 脚本里设定的output_dir是否已经存在,如果不存在，说明是第一次运行，直接跳过整个判断
    if os.path.isdir(training_args.output_dir):
        # 寻找名为checkpoint-XXXX 的子文件夹，并自动挑选数字最大的那一个（即最近的一次存档）
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        # get_last_checkpoint没找到存档（None），但文件夹里却有文件（len > 0）
        # 意味着你之前的实验没存好，或者是你把新实验指到了一个旧的、不相关的文件夹
        # 系统怕直接开始训练会把之前辛苦跑出来的其他数据给删了，所以抛出错误提示你手动清理
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"输出路径 ({training_args.output_dir}) 非空 "
            )
        # 到了之前的存档（not None），且用户在启动命令里没有显式指定从哪个特定位置恢复
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"从 {last_checkpoint}恢复训练"
            )
        
    # 设置随机数种子.
    set_seed(training_args.seed)

    """ 4------初始化模型"""
    
    # 从零开始训练模型
    if model_args.config_name is not None:
        # AutoConfig.from_pretrained()获取模型配置
        config = AutoConfig.from_pretrained(model_args.config_name)
        logger.warning("你正在从零初始化一个模型")
        logger.info(f"模型参数配置地址：{model_args.config_name}")
        logger.info(f"模型参数：{config}")

        # from_config根据config创建模型对象，模型的权重是随机生成的，没有任何知识
        model = AutoModelForCausalLM.from_config(config,trust_remote_code=True)

        # 计算模型参数总量（去重统计）
        # p.data_ptr() 用于处理参数共享的情况，确保不重复计算
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"预训练一个新模型 - Total size={n_params/2**20:.2f}M params")

    elif model_args.model_name_or_path is not None:
        logger.warning("你正在初始化一个预训练模型")
        logger.info(f"模型参数地址：{model_args.model_name_or_path}")
        # from_pretrained加载模型结构 + 加载已经训练好的权重
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path,trust_remote_code=True)
        
        # 计算模型参数总量
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"继承一个预训练模型 - Total size={n_params/2**20:.2f}M params")
    else:
        logger.error("config_name 和 model_name_or_path 不能均为空")
        raise ValueError("config_name 和 model_name_or_path 不能均为空")

    # 初始化 Tokenizer
    # model_args.tokenizer_name是在 sh 脚本中定义的地址
    # 如果是预训练：通常指向一个已经做好的分词器文件夹
    # 如果是微调：通常直接指向预训练模型的路径
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)
    logger.info("完成 tokenzier 加载")
    logger.info(f"tokenzier 配置地址：{model_args.tokenizer_name}")


    # 加载预训练数据
    '''
    train是一个DatasetDict类型的，大概类似下面这样，然后每一个feature又可以展开为列
    DatasetDict({
        train: Dataset({
            features: ['text', 'source', 'date'],
            num_rows: 2
        })
    })
    展开features: ['text', 'source', 'date']：
    text (string 列)	source (string 列)	date (string 列)
    0	"大语言模型如何工作？"	"知乎"	"2024-01-01"
    1	"Python 编程基础教程"	"博客"	"2024-02-15"

    所以ds["train"][0]在这里输出应该是第一条样本：
      0	"大语言模型如何工作？"	"知乎"	"2024-01-01"
    '''
    ds = load_dataset('json', data_files=data_args.train_files)
    logger.info("完成训练集加载")
    logger.info(f"训练集地址：{data_args.train_files}")
    logger.info(f'训练文件总数:{len(ds["train"])}')
    # logger.info(f"训练集采样：{ds["train"][0]}")


    # 文本 tokenize
    column_names = list(ds["train"].features)
    logger.info('训练集特征：', column_names)
    text_column_name = "text" if "text" in column_names else column_names[0]

    # tokenize 函数
    # examples到时候调用函数的时候会是一批text
    def tokenize_function(examples):
        output = tokenizer([item for item in examples[text_column_name]])
        # output：包含 input_ids 和 attention_mask 的字典
        return output

    # 仅主进程进行数据预处理，逻辑同s2_data_process
    # 在分布式训练中，脚本会同时启动多个进程，如果8个进程同时往同一个缓存文件里写数据，文件会直接崩掉
    # 主进程写完后，子进程直接读取现成的缓存即可
    # desc：一个描述标签
    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = ds.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset"
        )

    # 确定文本切块的block_size
    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "tokenizer 支持大于 1K 的上下文长度，默认设置为 1K"
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"设定的块长为 ({data_args.block_size}) ，大于模型的上下文长度"
                f"将块长设置为模型上下文长度：{tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # 逻辑同s2_data_process
    def group_texts(examples):
        # 将文本段拼接起来
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        # 计算拼起来的整体长度
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # 如果长度太长，进行分块
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }    
        result["labels"] = result["input_ids"].copy()
        return result
    
    # 利用map高效完成文本分块，逻辑同s2_data_process
    with training_args.main_process_first(desc="文本分块"):
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=True,
            desc=f"文本分块到{block_size}",
            batch_size = 40000,
        )
        logger.info("完成数据预处理")
        train_dataset = lm_datasets["train"]
    

    # 同s3_train_withTrainer中Trainer的初始化
    logger.info("初始化 Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset= IterableWrapper(train_dataset),
        tokenizer=tokenizer,
        data_collator=default_data_collator
    )

    # 从 checkpoint 加载
    checkpoint = None
    # 优先级 1：用户在命令行显式指定的路径
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    # 优先级 2：自动检测到的最新存档（之前代码找出的 last_checkpoint）
    elif last_checkpoint is not None:
            checkpoint = last_checkpoint

    logger.info("开始训练")
    # 如果resume_from_checkpoint是 None，模型从随机初始化（或加载的预训练权重）开始从第 0 步练
    # 如果这个值是路径，Trainer 会加载模型权重，恢复优化器状态（Optimizer State）、
    # 学习率调度器（Scheduler）以及数据采样器的进度
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # 训练过程中自动保存的 checkpoint-XXX 包含很多调试信息，文件巨大
    # 而trainer.save_model() 在训练结束时调用，
    # 把最干净、最完整的模型权重、配置文件和分词器文件保存在 output_dir 的根目录下
    # 执行完这一行后，文件夹里会出现 pytorch_model.bin（或 model.safetensors）、config.json 等
    trainer.save_model() 

if __name__ == "__main__":
    main()