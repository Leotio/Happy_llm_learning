'''预训练一个小型LLM'''
import tqdm
import json



# 1---处理预训练数据

def split_text(text, chunk_size=512):
    """将文本按照指定长度切分为块"""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

input_file = 'mobvoi_seq_monkey_general_open_corpus.jsonl'

# 以追加模式打开seq_monkey_datawhale.jsonl作为输出文件,处理后的文本块写入这个文件
# .jsonl文件中的每一行都是一个独立的JSON对象,易于按行处理
with open('seq_monkey_datawhale.jsonl', 'a', encoding='utf-8') as pretrain:
    # 以只读模式打开之前指定的 input_file（原始数据集）
    with open(input_file, 'r', encoding='utf-8') as f:
        # 将输入文件中的所有行（每个JSON对象一行）读取到一个列表中
        data = f.readlines()
        for line in tqdm(data, desc=f"Processing lines in {input_file}", leave=False):
            # 将当前行（一个 JSON 格式的字符串）解析成Python字典
            line = json.loads(line)
            # 从解析后的字典中，提取键为 'text' 的值，即原始文本内容
            text = line['text']
            # 调用前面的split_text函数，将提取出的长文本切分成 512 个字符长度的文本块列表
            chunks = split_text(text)
            for chunk in chunks:
                # 将每个文本块封装回一个新的 JSON 对象，格式为 {'text': chunk}
                # json.dumps 将字典转换为JSON字符串。
                # ensure_ascii=False 确保中文字符不被转义
                pretrain.write(json.dumps({'text': chunk}, ensure_ascii=False) + '\n')

# 2---处理SFT数据

# 处理BelleGroup/train_3.5M_CN.json原始数据集
def convert_message(data):
    """
    将原始数据转换为标准格式
    """
    # 对话的开头添加一个标准的 System Message，用于设置 AI 的行为、角色或限制
    message = [
        {"role": "system", "content": "你是一个AI助手"},
    ]
    for item in data:
        # 检查当前轮次的发送者（'from' 键）是否是 'human'
        if item['from'] == 'human':
            # 如果是则将其转换为标准格式：'role' 设为 'user'，'content' 为 'value' 键对应的内容
            message.append({'role': 'user', 'content': item['value']})
        elif item['from'] == 'assistant':
            message.append({'role': 'assistant', 'content': item['value']})
    return message

# 以追加模式 ('a') 打开输出文 BelleGroup_sft.jsonl(存储 SFT 数据的最终文件)
with open('BelleGroup_sft.jsonl', 'a', encoding='utf-8') as sft:
    
    # 只读模式 ('r') 打开原始的 BelleGroup 数据文件
    with open('BelleGroup/train_3.5M_CN.json', 'r', encoding='utf-8') as f:
        data = f.readlines()

        for item in tqdm(data, desc="Processing", unit="lines"):
            # 将当前行的JSON字符串解析成 Python字典item
            item = json.loads(item)
            # 调用前面定义的 convert_message 函数，传入原始对话列表，获得标准格式的 message 列表
            message = convert_message(item['conversations'])
            # 将转换后的标准 message 列表转换回 JSON 字符串
            sft.write(json.dumps(message, ensure_ascii=False) + '\n')



