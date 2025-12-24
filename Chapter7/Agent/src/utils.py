'''
    将tools.py里面的工具转为特定的JSON Schema格式，OpenAI API才能理解这些工具
'''
import inspect
from datetime import datetime
import pprint

def function_to_json(func) -> dict:
    # 定义 Python 类型到 JSON 数据类型的映射
    type_map = {
        str: "string",       # 字符串类型映射为 JSON 的 "string"
        int: "integer",      # 整型类型映射为 JSON 的 "integer"
        float: "number",     # 浮点型映射为 JSON 的 "number"
        bool: "boolean",     # 布尔型映射为 JSON 的 "boolean"
        list: "array",       # 列表类型映射为 JSON 的 "array"
        dict: "object",      # 字典类型映射为 JSON 的 "object"
        type(None): "null",  # None 类型映射为 JSON 的 "null"
    }


    try:
        # inspect.signature(func)：这是 Python 标准库的强大工具
        # 它能“窥探”函数内部，看它有多少个参数、参数叫什么、有没有默认值、有没有类型注解
        signature = inspect.signature(func)

    except ValueError as e:
        raise ValueError(
            f"无法获取函数 {func.__name__} 的签名: {str(e)}"
        )

    # 用于存储参数信息的字典
    parameters = {}
    for param in signature.parameters.values():
        try:
            # param.annotation：获取在定义函数时写的类型标注（例如 query: str 中的 str）
            # 如果无法找到对应的类型则默认设置为 "string"
            param_type = type_map.get(param.annotation, "string")

        except KeyError as e:
            # 如果参数类型不在 type_map 中，抛出异常并显示具体错误信息
            raise KeyError(
                f"未知的类型注解 {param.annotation}，参数名为 {param.name}: {str(e)}"
            )
        # 将参数名及其类型信息添加到参数字典中
        parameters[param.name] = {"type": param_type}

    # 获取函数中所有必需的参数（即没有默认值的参数）
    required = [
        param.name
        for param in signature.parameters.values()
        # inspect._empty是 inspect 库用来表示“这个参数没有默认值”的特殊占位符
        if param.default == inspect._empty
    ]

    # 返回包含函数描述信息的字典
    # 这个嵌套格式是 OpenAI 指定的标准格式
    return {
        "type": "function",
        "function": {
            "name": func.__name__,            # 函数名称
            "description": func.__doc__ or "", # 函数docstring，如果不存在则为空字符串
            "parameters": {
                "type": "object",
                "properties": parameters,     # 上面生成的parameters字典
                "required": required,         # 上面生成的必须参数的列表
            },
        },
    }
