from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

chat_model = ChatOpenAI(
    openai_api_key=r"sk-lolzalmconplprhxshsxunwdhpvlflbdbhneyyuyboqfbtsk",
    base_url=r"https://api.siliconflow.cn/v1",
    model=r"Qwen/Qwen2.5-7B-Instruct",
)

"""invoke() 同步调用模型生成回复"""
# 输入字符串
response = chat_model.invoke("你好吗？我很好。")
print(response.content)
# 输入消息列表
messages = [
    SystemMessage(content="你是一个爱使用颜文字的个人助手"),
    HumanMessage(content="你好吗？我很好。")
]
response = chat_model.invoke(messages)
print(response.content)

"""generate() 异步调用模型生成回复"""

inputs = [
    [
        SystemMessage(content="你是一个爱使用颜文字的个人助手"),
        HumanMessage(content="1+1=?")
    ],
    [
        SystemMessage(content="你一个严肃冷静温柔的家庭教师"),
        HumanMessage(content="1+1=?")
    ]
]

result = chat_model.generate(inputs)
# 遍历打印回复
for res in result.generations:
    print(res[0].text)

"""stream() 流式响应"""
stream_response = chat_model.stream("请说明叶修为什么是荣耀之神")
for chunk in stream_response:
    # flush=True 确保立即打印，而不是等待print缓冲区满
    print(chunk.content, end="", flush=True)

"""predict() 和 predict_messages()"""
# 直接输出字符串
reply = chat_model.predict("你好吗？我很好。")
print(reply)  

print(f"\n\n\n")

# 输出AIMessage
reply = chat_model.predict_messages([HumanMessage(content="你好吗？我很好。")])
print(reply.content)  

"""batch() 简单批量调用"""
inputs = [
    "写一个小红书美妆标题",  # 直接传字符串
     [SystemMessage(content="你是美食博主"), HumanMessage(content="推荐1道快手菜")],  # 单组消息
    "用3个词形容秋天"  # 直接传字符串
]

# 批量调用
results = chat_model.batch(inputs)
# 遍历结果（与输入顺序一致）
for res in results:
    print(res.content)