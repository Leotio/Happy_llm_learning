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
# 生成器函数使用yield关键字而不是 return 来返回值
# str：生成器每次生成的数据类型
# None：发送的值，外部可以发送回生成器的值类型
# None：返回值，函数终止时（yield 完毕后）返回的值
def read_texts_from_jsonl(file_path: str) -> Generator[str, None, None]:
    with open(file_path, 'r', encoding='utf-8') as f:
        # enumerate(iterable, start=0)：start指定索引开始的值
        for line_num, line in enumerate(f, 1):
            try:

                # 只有当外部代码(不是这里的for的代码，是到时候比如
                # generator = read_texts_from_jsonl(path)
                # for text in generator:  ）
                # 一个训练循环需要下一段文本时，函数才会运行一行 json.loads(line)

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
def create_tokenizer_config(save_dir: str) -> None:
    config = {
        # 控制 Tokenizer在编码时是否自动在序列开头和结尾添加这些标记
        # 此处设为 False，意味着由chat_template来控制添加特殊标记
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,

        # eos_token用于指示序列的终止,例如，模型在生成序列时，遇到eos_token后就会停止生成
        # 更具体地就是遇到<|im_end|>时停止
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "pad_token": "<|im_end|>",

        # 未知标记,用于替换那些在词汇表中不存在的词汇
        "unk_token": "<unk>",
        # 模型能够处理的最大序列长度,设置一个非常大的值通常表示没有硬性限制
        "model_max_length": 1000000000000000019884624838656,
        # 禁用清理：将保留所有 Token 之间连接时引入的空格，不做任何额外的空格清理或调整
        "clean_up_tokenization_spaces": False,
        # 指定Tokenizer类型,PreTrainedTokenizerFast 表示这是一个基于 Rust 编写的高性能Tokenizer
        "tokenizer_class": "PreTrainedTokenizerFast",

        # 将非结构化的对话历史数据转换为结构化的、可理解的 Token 序列
        "chat_template": (
            # 每个元素 message 都是一个字典，通常包含 {'role': '...', 'content': '...'} 两个键
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'user' %}"
            "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'assistant' %}"
            "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        )
    }

    # 保存主配置⽂件
    with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        # 使用 json.dump 将字典 config 写入指定路径下的 tokenizer_config.json 文件中
        json.dump(config, f, ensure_ascii=False, indent=4)

    # 创建special_tokens_map.json
    # 这个字典是 tokenizer_config.json中定义的核心特殊Token的冗余映射，用于库的快速查找和兼容性
    special_tokens_map = {
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "unk_token": "<unk>",
        "pad_token": "<|im_end|>",
        # 定义了除了 BOS/EOS/UNK/PAD 之外，还需要在词汇表中添加的其他特殊 Token，
        # 例如传统的句子开始 <s> 和句子结束 </s> 标记（尽管这里主要使用 <|im_start|> 和 <|im_end|>)
        "additional_special_tokens": ["<s>", "</s>"]
    }

    with open(os.path.join(save_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, ensure_ascii=False, indent=4)
 

'''训练BPE Tokenizer'''
# 定义一个训练函数，用于训练Tokenizer并保存训练好的Tokenizer文件

# 此处使用tokenizers库中的Tokenizer类来实现
def train_tokenizer(data_path: str, save_dir: str, vocab_size: int = 8129) -> None:
    os.makedirs(save_dir, exist_ok=True)

    # -----初始化tokenizer-----

    # 指定Tokenizer的核心算法模型为Byte-Pair Encoding (BPE)
    # unk_token="<unk>"：在分词过程中，如果遇到一个在词汇表中不存在的词，用<unk>标记替换;
    # 这样子更安全，虽然字节级别的编码很少出现不存在的
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # 定义在分词之前对输入文本进行标准化的规则 NFKC()，
    # NFKC主要用于将字符转换为它们的规范形式，例如，将全角字符转换为半角字符，
    # 确保 Tokenizer 不会因为同一个字符有不同的 Unicode 表示形式而将其视为不同的词汇
    tokenizer.normalizer = NFKC()

    # 在 BPE 算法运行 之前，采用字节级别将原始字符串初步切割成更小的单元。
    # 将文本的每一个字符映射到其字节表示上，意味着 Tokenizer 的词汇表将以字节对为基础进行构建
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # 分词之后，采用与预分词器对应的字节级别解码将Token ID序列转换回可读字符串（即 解码）
    tokenizer.decoder = decoders.ByteLevel()

    # -----特殊配置token-----
    # 确保不会被BPE算法分解，并在最终的词汇表中拥有独立的、唯一的Token ID
    special_tokens = [
        "<unk>", 
        "<s>", 
        "</s>", 
        "<|im_start|>", 
        "<|im_end|>"
    ]


    # -----配置训练器-----

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens = special_tokens,
        # 当一个子词对（或基础 Token）在训练语料中出现的次数大于或等于2时，才会被考虑加入到最终的词汇表中
        # 子词对：Token "l" 和 "o" 经常相邻出现，BPE 就会创建一个新的子词 Token "lo"
        min_frequency=2,
        # 进度显示，在训练过程中，终端会打印出进度条或其他反馈信息
        show_progress=True,
        # 初始字母表：确保BPE算法从所有256个字节（ByteLevel预分词器定义的全部字符集）作为初始单元开始训练
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )


    # -----训练tokenizer-----

    print(f"Training tokenizer with data from {data_path}")
    # 用之前定义的read_texts_from_jsonl返回一个可迭代对象texts，包含所有待用于训练的文本字符串
    texts = read_texts_from_jsonl(data_path)

    # train_from_iterator设计为从一个迭代器中流式读取数据
    # trainer：预配置的BpeTrainer对象，指定训练过程中使用的规则和约束，
    #  length：数据文件的大小
    tokenizer.train_from_iterator(texts, trainer=trainer, length=os.path.getsize(data_path))

    # -----验证特殊token映射-----
    # special_tokens参数将这些标记保留，Tokenizers会分配最小、固定的I。
    # 代码用于强制确认这些 ID 分配与模型训练时的预期完全一致（按照special_tokens定义的时候的顺序分配ID）
    try:
        assert tokenizer.token_to_id("<unk>") == 0
        assert tokenizer.token_to_id("<s>") == 1
        assert tokenizer.token_to_id("</s>") == 2
        assert tokenizer.token_to_id("<|im_start|>") == 3
        assert tokenizer.token_to_id("<|im_end|>") == 4
    except AssertionError as e:
        print("Special tokens mapping error:", e)
        raise

    # -----保存tokenizer文件-----

    # 将Tokenizer对象序列化并保存到文件的核心方法,tokenizer.json是保存Tokenizer所有训练结果的标准文件名
    # 它包含了 Tokenizer 的全部核心信息：模型/词汇表/规则配置
   
    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))

    # -----创建配置文件-----

    # 自定义的函数,创建tokenizer_config.json辅助配置文件
    # 包含了 Hugging Face Transformers 库加载 Tokenizer 时所需的所有元数据和特殊标记映射
    # 告诉 Transformers 库用这个类，以这些特殊标记和配置来加载 tokenizer.json 文件
    create_tokenizer_config(save_dir)
    print(f"Tokenizer saved to {save_dir}")


    # "tokenizer.json"包含所有底层数据和逻辑,它的特点是大且固定
    # "tokenizer_config.json"包含高级 API 需要知道的所有配置和元数据,它的特点是小且可变


'''使用训练好的Tokenizer'''
def eval_tokenizer(tokenizer_path: str) -> None:
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return
    
    # 测试基本属性
    