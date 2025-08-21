# 第三方平台的API调用
from openai import OpenAI

client = OpenAI(
    api_key="sk-da5d4ca868784ed79564a2302eed9be7",    # API Key
    base_url="https://api.deepseek.com"    # 请求地址
)

response = client.chat.completions.create(
    model="deepseek-chat",    # 模型名称
    messages=[
        {"role": "system", "content": "你是一个资深的古诗文研究学者"},
        {"role": "user", "content": "请简要介绍一下诗鬼李贺"},
    ],
    stream=False    # 非流式：需要等待模型全部生成完成
)

print(response.choices[0].message.content)