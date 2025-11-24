import requests  # 导入requests库，用于发送HTTP请求
import json

# 定义请求地址
url = "https://api.siliconflow.cn/v1/chat/completions"

# 设置请求头，包含API密钥和请求内容类型
headers = {
    "Authorization": "Bearer sk-lolzalmconplprhxshsxunwdhpvlflbdbhneyyuyboqfbtsk",
    "Content-Type": "application/json"
}

# 定义请求数据
data = {
    "model": "Qwen/Qwen2.5-72B-Instruct",
    "messages": [
        {"role": "system", "content": "你是一个爱使用颜文字的生活助手。"},
        {"role": "user", "content": "你好吗？我很好。"}
    ],
    "temperature": 0.5,  # 温度（0-1，默认0.5），值越高越随机
    "top_p": 0.9,  # 核采样（0-1，默认0.9），候选词的累积概率：高=候选多，低=候选少（高概率词）
    "max_tokens": 512,  # 最大生成长度，默认1024
    "stream": False,  # 是否流式返回
}

# 发送POST请求
response = requests.post(url, headers=headers, data=json.dumps(data))
# 打印响应数据
print(response.json())
