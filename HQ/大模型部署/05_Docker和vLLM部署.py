from openai import OpenAI

# 创建客户端实例
# 如果你的Python代码运行在Docker容器所在的同一台机器上，使用localhost
client = OpenAI(
    base_url="http://localhost:8000/v1",  # 注意端口是8000，不是8008
    api_key="EMPTY"  # 可以设置为"EMPTY"或任意非空字符串，但不能是"none"
)

# 调用聊天接口
try:
    response = client.chat.completions.create(
        model="/models",  # 需要是创建容器时，指定的映射，即加载模型的位置
        messages=[
            {"role": "system", "content": "你是可爱的个人助手"},
            {"role": "user", "content": "你好吗？我很好。"}

        ],
        stream=False,
        max_tokens=1000  # 可选：限制生成的最大token数量
    )

    # 打印回答
    print("模型回复：", response.choices[0].message.content)

except Exception as e:
    print(f"请求出错: {e}")