from openai import OpenAI

# 关键：将 base_url 指向你的服务器地址
# Ubuntu系统的IP地址(在Ubuntu系统中输入hostname-I命令查看)
client = OpenAI(
    base_url="http://192.168.144.217:8008/v1",
    api_key="none"
)

# 调用聊天接口
response = client.chat.completions.create(
    model="/mnt/c/HuggingFace/Qwen3-0.6B",
    messages=[
        {"role": "user", "content": "你好吗？"}
    ],
    stream=False,
)

# 打印回答
print("模型回复：", response.choices[0].message.content)