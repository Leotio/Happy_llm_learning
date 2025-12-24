import datetime
import wikipedia
import requests



"""
Agent 外部工具（Tools）
在 Agent 的运行流程中，这些函数会经历以下三个阶段：
声明阶段：框架（如 LangChain）读取这些函数的 Docstring。
         比如它看到 :param latitude: 纬度坐标，就知道如果用户问“上海天气”，它需要先去查上海的经纬度，然后传给这个参数。
决策阶段：当用户问“现在北京多少度？”时，LLM 发现自己没有实时数据，
         但看到 get_current_temperature 的描述符合需求，于是输出一个指令：call: get_current_temperature(lat=39.9, lon=116.4)。
执行与反馈:Python 运行上面的函数，拿到“15°C”，再把这个结果喂回给 LLM。
          LLM 最后组织语言回答用户：“北京现在的温度是 15 摄氏度。”
"""
# 获取当前日期和时间
def get_current_datetime() -> str:
    """
    获取真实的当前日期和时间。
    :return: 当前日期和时间的字符串表示。
    """
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_datetime

def add(a: float, b: float):
    """
    计算两个浮点数的和。
    :param a: 第一个浮点数。
    :param b: 第二个浮点数。
    :return: 两个浮点数的和。
    """
    return str(a + b)

def mul(a: float, b: float):
    """
    计算两个浮点数的积。
    :param a: 第一个浮点数。
    :param b: 第二个浮点数。
    :return: 两个浮点数的积。
    """
    return str(a * b)

def compare(a: float, b: float):
    """
    比较两个浮点数的大小。
    :param a: 第一个浮点数。
    :param b: 第二个浮点数。
    :return: 比较结果的字符串表示。
    """
    if a > b:
        return f'{a} is greater than {b}'
    elif a < b:
        return f'{b} is greater than {a}'
    else:
        return f'{a} is equal to {b}'

def count_letter_in_string(a: str, b: str):
    """
    统计字符串中某个字母的出现次数。
    :param a: 要搜索的字符串。
    :param b: 要统计的字母。
    :return: 字母在字符串中出现的次数。
    """
    string = a.lower()
    letter = b.lower()
    
    count = string.count(letter)
    return(f"The letter '{letter}' appears {count} times in the string.")

def search_wikipedia(query: str) -> str:
    """
    在维基百科中搜索指定查询的前三个页面摘要。
    :param query: 要搜索的查询字符串。
    :return: 包含前三个页面摘要的字符串。
    """
    page_titles = wikipedia.search(query)
    summaries = []
    for page_title in page_titles[: 3]:  # 取前三个页面标题
        try:
            # 使用 wikipedia 模块的 page 函数，获取指定标题的维基百科页面对象。
            wiki_page = wikipedia.page(title=page_title, auto_suggest=False)
            # 获取页面摘要
            summaries.append(f"页面: {page_title}\n摘要: {wiki_page.summary}")
        except (
                wikipedia.exceptions.PageError,
                wikipedia.exceptions.DisambiguationError,
        ):
            pass
    if not summaries:
        return "维基百科没有搜索到合适的结果"
    return "\n\n".join(summaries)


def get_current_temperature(latitude: float, longitude: float) -> str:
    """
    获取指定经纬度位置的当前温度。
    :param latitude: 纬度坐标。
    :param longitude: 经度坐标。
    :return: 当前温度的字符串表示。
    """

    # Open Meteo API 的URL
    # Open-Meteo API:一个免费的公开天气接口,来获取精准的实时温度
    open_meteo_url = "https://api.open-meteo.com/v1/forecast"

    # 请求参数
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'hourly': 'temperature_2m',
        'forecast_days': 1,
    }

    # 发送 API 请求
    response = requests.get(open_meteo_url, params=params)

    # 检查响应状态码
    if response.status_code == 200:
        # 解析 JSON 响应
        results = response.json()
    else:
        # 处理请求失败的情况
        raise Exception(f"API Request failed with status code: {response.status_code}")

    # 获取当前 UTC 时间
    current_utc_time = datetime.datetime.now(datetime.UTC)

    # 将时间字符串转换为 datetime 对象
    time_list = [datetime.datetime.fromisoformat(time_str).replace(tzinfo=datetime.timezone.utc) for time_str in
                 results['hourly']['time']]

    # 获取温度列表
    temperature_list = results['hourly']['temperature_2m']

    # 找到最接近当前时间的索引
    # min()函数里面可以使用key参数，这样子就会比较 将key作为索引 对应的真实值的大小
    # 带 key 的行为：会把 range 里的每一个索引 i 丢进 key 函数里计算，然后根据计算结果的大小来选出最小的 i
    closest_time_index = min(range(len(time_list)), key=lambda i: abs(time_list[i] - current_utc_time))

    # 获取当前温度
    current_temperature = temperature_list[closest_time_index]

    # 返回当前温度的字符串形式
    return f'现在温度是 {current_temperature}°C'