import time
from openai import OpenAI

def stream_demo():
    def stream(n):
        for i in range(n):
            time.sleep(1)
            yield i
    for i in stream(5):
        print(i)    # 一个词一个词生成，每个词之间间隔1秒
    print("流式推理完成")

def docker_vllm_stream_demo():
    client = OpenAI(
        api_key="EMPTY",    # APIKey
        base_url="http://192.168.144.217:8008/v1",  # 请求地址  /chat/completions
    )

    # 设置生成参数和输入消息
    gen_kwargs = {
        "max_tokens": 512,              # 生成的最大长度
        "temperature": 0.7,             # 生成丰富性，越大越有创造力 越小越确定
        "top_p": 0.8,                   # 采样时的前P个候选词，越大越随机
        "extra_body": {
            "do_sample": True,          # 是否使用概率采样
            "top_k": 50,                # 采样时的前K个候选词，越大越随机
            "repetition_penalty": 1.2,  # 重复惩罚系数，越大越不容易重复
        }
    }

    # 面向对象的模块化编程
    response = client.chat.completions.create(
        model="/mnt/C/Huggingface/Qwen3-0.6B",  # 使用的模型：可以是模型名称 或者 模型路径
        messages=[
            {"role": "system", "content": "你是一位心理咨询师"},
            {"role": "user", "content": "请问治疗失眠常用药物有哪些？"},
        ],
        stream=False,
        **gen_kwargs,
    )
    # Python里面的数据类型的知识点
    print(response.choices[0].message.content)






if __name__ == '__main__':
    # stream_demo()
    docker_vllm_stream_demo()
