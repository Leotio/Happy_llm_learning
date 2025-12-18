'''使用transfromer提供的Trainer类来进行训练'''

from transformers import TrainingArguments
from transformers import Trainer, default_data_collator
from torchdata.datapipes.iter import IterableWrapper
from s1_init_llm import tokenizer,model
from s2_data_process import train_dataset

# 训练参数
training_args = TrainingArguments(
    output_dir="output",# 训练参数输出路径
    per_device_train_batch_size=4,# 训练的 batch_size
    gradient_accumulation_steps=4,# 梯度累计步数，实际 bs = 设置的 bs * 累计步数
    logging_steps=10,# 打印 loss 的步数间隔
    num_train_epochs=1,# 训练的 epoch 数
    save_steps=100, # 保存模型参数的步数间隔
    learning_rate=1e-4,# 学习率
    gradient_checkpointing=True# 开启梯度检查点
)

# 基于初始化的model、tokenizer和training_args，处理好的训练数据集，实例化Trainer对象
trainer = Trainer(
    model=model,
    args=training_args,
    # IterableWrapper：用于处理流式数据集,数据集非常大，
    # 无法一次性加载进内存，这种封装允许模型像接水管一样，边读边练
    train_dataset = IterableWrapper(train_dataset),
    # 设置为 None 表示在训练过程中不进行周期性的验证评分
    eval_dataset=None,
    tokenizer=tokenizer,
    # 数据整理器:
    # 从 train_dataset 中取出长度不一的多个样本组成一个 Batch 时，data_collator 负责将它们对齐
    # default_data_collator：将 Batch 中的数据转换为标准张量（Tensors）
    data_collator=default_data_collator,
)

# Trainer 是 Hugging Face 提供的高级 API，封装了 PyTorch 原生的 
# forward 、backward（反向传播）、权重更新、梯度裁剪、分布式训练等，只需关注配置而非底层代码
trainer.train()