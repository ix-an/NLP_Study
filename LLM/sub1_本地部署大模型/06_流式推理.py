from openai import OpenAI

def docker_vllm_stream_demo():
    client = OpenAI(
        api_key="EMPTY",    # APIKey
        base_url="http://localhost:8000/v1",  # 请求地址  /chat/completions
    )

    # 设置生成参数和输入消息
    gen_kwargs = {
        "max_tokens": 2048,              # 生成的最大长度
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
        model="/models",  # 需要是创建容器时，--model 指定的容器下目录
        messages=[
            {"role": "system", "content": "你是一位心理咨询师"},
            {"role": "user", "content": "请问治疗失眠常用药物有哪些？"},
        ],
        stream=True,  # 启动流式推理
        **gen_kwargs,
    )

    # 打印流式回复
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content is not None:
            print(content, end="", flush=True)


if __name__ == '__main__':
    docker_vllm_stream_demo()
