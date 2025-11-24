from openai import OpenAI

api_key = "sk-lolzalmconplprhxshsxunwdhpvlflbdbhneyyuyboqfbtsk"
base_url = "https://api.siliconflow.cn/v1"  # 注意和request中的url不一样

client = OpenAI(
    api_key=api_key,  # API Key
    base_url=base_url,  # 请求地址
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-72B-Instruct",    # 模型名称
    messages=[
        {"role": "system", "content": "你是一个爱用颜文字的古诗文科普博主"},
        {"role": "user", "content": "请简要介绍一下诗鬼李贺"},
    ],
    max_tokens=512,
    stream=False
)

print(response.choices[0].message.content)
# for chunk in response:
#     print(chunk.choices[0].delta.content)
