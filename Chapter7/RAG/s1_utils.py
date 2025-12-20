'''加载文档并进行切分'''

import os
from typing import Dict, List, Optional, Tuple, Union

import PyPDF2
import markdown
import json
from tqdm import tqdm
import tiktoken
from bs4 import BeautifulSoup
import re

enc = tiktoken.get_encoding("cl100k_base")


class ReadFiles:
    """
    class to read files
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self.file_list = self.get_files()

    # 在指定的文件夹及其子文件夹里，找出所有的 .md, .txt, .pdf 文件
    def get_files(self):
        file_list = []
        for filepath, dirnames, filenames in os.walk(self._path):
            # os.walk 函数将递归遍历指定文件夹
            for filename in filenames:
                # 通过后缀名判断文件类型是否满足要求
                if filename.endswith(".md"):
                    # 如果满足要求，将其绝对路径加入到结果列表
                    # os.path.join 把文件夹路径和文件名拼成一个完整的绝对路径
                    # 比如把 "/data" 和 "help.md" 拼成 "/data/help.md"
                    file_list.append(os.path.join(filepath, filename))
                elif filename.endswith(".txt"):
                    file_list.append(os.path.join(filepath, filename))
                elif filename.endswith(".pdf"):
                    file_list.append(os.path.join(filepath, filename))
        return file_list


    # 把文件从硬盘读进内存，切碎，然后汇总
    def get_content(self, max_token_len: int = 600, cover_content: int = 150):
        docs = []
        # 读取文件内容
        for file in self.file_list:
            # read_file_content定义在后面，根据后缀选择不同读取方式
            content = self.read_file_content(file)
            chunk_content = self.get_chunk(
                content, max_token_len=max_token_len, cover_content=cover_content)
            docs.extend(chunk_content)
        return docs
    

    # --------对文档进行切分--------

    # @classmethod代表是类方法，不需要进行实例化，可以直接通过类名.的方法来调用
    @classmethod
    def get_chunk(cls, text: str, max_token_len: int = 600, cover_content: int = 150):
        """
        cover_content:重叠内容的长度
        max_token_len:每个数据块允许的最大 Token 长度
        """
        # 存储最终切分好的所有文本块
        chunk_text = []

        # 当前正在构建的块的已用 Token 长度
        curr_len = 0
        # 当前正在构建的块的文本内容
        curr_chunk = ''

        # 实际有效的新内容长度 = 最大长度 - 覆盖长度
        token_len = max_token_len - cover_content
        # 按行切分原始文本
        lines = text.splitlines()  

        for line in lines:
            # 保留空格，只移除行首行尾空格
            line = line.strip()
            # 计算这一行有多少个 Token
            line_len = len(enc.encode(line))
            
            if line_len > max_token_len:
                # 如果单行长度就超过限制，则将其分割成多个块
                
                if curr_chunk:
                    chunk_text.append(curr_chunk) # 先保存当前块
                    curr_chunk = '' # 然后清空
                    curr_len = 0
                
                # 将超长行转为 Token 序列
                line_tokens = enc.encode(line)

                # 计算这一行需要被切成多少个小块
                num_chunks = (len(line_tokens) + token_len - 1) // token_len
                
                for i in range(num_chunks):
                    start_token = i * token_len
                    end_token = min(start_token + token_len, len(line_tokens))
                    
                    # 取出对应的 Token 片段
                    chunk_tokens = line_tokens[start_token:end_token]
                    # 解码token片段回文本
                    chunk_part = enc.decode(chunk_tokens)
                    
                    # 如果不是第一块，且之前已经有块了，就加上“覆盖内容”
                    if i > 0 and chunk_text:
                        prev_chunk = chunk_text[-1]
                        # 取前一个块的最后一部分作为当前块的开头
                        cover_part = prev_chunk[-cover_content:] if len(prev_chunk) > cover_content else prev_chunk
                        chunk_part = cover_part + chunk_part
                    
                    chunk_text.append(chunk_part)
                
                # 重置当前块状态
                curr_chunk = ''
                curr_len = 0
                
            elif curr_len + line_len + 1 <= token_len:  # +1 for newline
                # 当前行可以加入当前块
                if curr_chunk:
                    curr_chunk += '\n'# 块内换行
                    curr_len += 1 # 长度记 1（换行符）
                curr_chunk += line
                curr_len += line_len
            else:
                # 当前行无法加入当前块，开始新块
                if curr_chunk:
                    chunk_text.append(curr_chunk)
                
                # 开始新块，添加覆盖内容
                if chunk_text:
                    prev_chunk = chunk_text[-1]
                    cover_part = prev_chunk[-cover_content:] if len(prev_chunk) > cover_content else prev_chunk
                    curr_chunk = cover_part + '\n' + line
                    curr_len = len(enc.encode(cover_part)) + 1 + line_len
                else:
                    curr_chunk = line
                    curr_len = line_len

        # 添加最后一个块（如果有内容）
        if curr_chunk:
            chunk_text.append(curr_chunk)

        return chunk_text

    @classmethod
    def read_file_content(cls, file_path: str):
        # 根据文件扩展名选择不同的读取方法
        if file_path.endswith('.pdf'):
            return cls.read_pdf(file_path)
        elif file_path.endswith('.md'):
            return cls.read_markdown(file_path)
        elif file_path.endswith('.txt'):
            return cls.read_text(file_path)
        else:
            raise ValueError("Unsupported file type")

    @classmethod
    def read_pdf(cls, file_path: str):
        # 以“二进制只读”模式打开PDF文件
        with open(file_path, 'rb') as file:
            # 创建一个 PDF 阅读器对象
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()
            return text

    @classmethod
    def read_markdown(cls, file_path: str):
        # 先转成 HTML 再提取纯文本，方便剔除 Markdown 的格式符号（如 **, # 等）
        with open(file_path, 'r', encoding='utf-8') as file:
            md_text = file.read()
            # 将 Markdown 语法转化为 HTML 语法（例如 # 转为 <h1>）
            html_text = markdown.markdown(md_text)
            # 使用BeautifulSoup从HTML中提取纯文本
            soup = BeautifulSoup(html_text, 'html.parser')
            plain_text = soup.get_text()
            # 使用正则表达式移除网址链接
            text = re.sub(r'http\S+', '', plain_text) 
            return text

    @classmethod
    def read_text(cls, file_path: str):
        # 读取文本文件
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

# 主要就是为了方便直接处理json文件
class Documents:
    """
        获取已分好类的json格式文档
    """
    def __init__(self, path: str = '') -> None:
        self.path = path
    
    def get_content(self):
        with open(self.path, mode='r', encoding='utf-8') as f:
            content = json.load(f)
        return content