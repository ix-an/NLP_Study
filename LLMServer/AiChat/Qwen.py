from openai import OpenAI

class Qwen:
    def __init__(self, host, model="/models"):
        self.client = OpenAI(
        api_key="EMPTY",    # APIKey
        base_url=host + "/v1",  # 请求地址  /chat/completions
        )
        self.gen_kwargs = {
            "max_tokens": 2048,              # 生成的最大长度
            "temperature": 0.7,             # 生成丰富性，越大越有创造力 越小越确定
            "top_p": 0.8,                   # 采样时的前P个候选词，越大越随机
            "extra_body": {
                "do_sample": True,          # 是否使用概率采样
                "top_k": 50,                # 采样时的前K个候选词，越大越随机
            }
        }
        self.model = model    # 模型


    def inference(self, messages, stream=False):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=stream,
            **self.gen_kwargs,
        )
        # 非流式回复
        if not stream:
            return response.choices[0].message.content

        # 流式回复
        for chunk in response:
            yield f'data:{chunk.choices[0].delta.content}'
        # 结束标记
        yield 'data:[DONE]'







