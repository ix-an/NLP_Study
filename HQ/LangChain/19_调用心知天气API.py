import requests  # 网络请求工具
from pydantic import Field


# 定义心知天气API的工具类
class WeatherTool:
    city: str = Field(description="城市名称")

    def __init__(self, api_key) -> None:
        self.api_key = api_key

    def run(self, city: str):
        city = city.split("\n")[0]  # 清除多余换行符
        url = (
            f"https://api.seniverse.com/v3/weather/now.json?"
            f"key={self.api_key}&"
            f"location={city}&"
            f"language=zh-Hans&"
            f"unit=c"
        )  # 括号内自动拼接字符串

        # 构建API请求
        response = requests.get(url)
        if response.status_code != 200:
            return "请求失败"

        # 解析JSON响应
        data = response.json()
        # 提取天气信息
        if data["results"]:
            weather = data["results"][0]["now"]
            return f"{city}的当前天气是：{weather['text']}，温度是：{weather['temperature']}℃"
        else:
            return f"无法获取{city}的天气信息"


API_KEY = "Sa_TQtZ7nZ32nVtMd"
weather_tool = WeatherTool(API_KEY)
print(weather_tool.run("重庆"))
print(weather_tool.run("扬州"))
