"""
多轮对话基本思路：
定义一个消息列表，用于存储对话历史，通过API接口向 OpenAI发送请求。
每次请求都包含完整的对话历史，OpenAI会根据上下文生成回答。
返回结果之后，将其添加到消息列表中，以便维持上下文。
"""

from openai import OpenAI

API_KEY = "sk-lolzalmconplprhxshsxunwdhpvlflbdbhneyyuyboqfbtsk"
URL = "https://api.siliconflow.cn/v1"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# 创建OpenAI对象
client = OpenAI(api_key=API_KEY,  base_url=URL)

# 初始化消息列表
messages = [
    {"role": "system", "content": "你是一个爱使用颜文字的私人助手"}
]

# 调用大模型的函数
def get_response(messages):
    """把 messages 发给 model → 拿到回复"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        return None

# 进行多轮对话
while True:
    # 获取用户输入
    user_input = input("询问任何问题：")
    if user_input.lower() == 'exit':
        print("对话结束")
        break

    # 将用户输入添加到消息历史
    messages.append({"role": "user", "content": user_input})

    # 获取模型回答
    model_response = get_response(messages)
    if model_response:
        print(f"🦋：{model_response}")
        # 将模型回答添加到消息历史
        messages.append({"role": "assistant", "content": model_response})
    else:
        print("私密马赛，这个问题暂时无法回答")

