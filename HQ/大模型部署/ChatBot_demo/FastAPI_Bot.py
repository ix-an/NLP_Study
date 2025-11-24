from fastapi import FastAPI, Body
from openai import AsyncOpenAI  # 异步OpenAI客户端
from typing import List
from fastapi.responses import StreamingResponse


# 初始化 FastAPI 应用
app = FastAPI(title="大模型多轮对话 API")

# 初始化 OpenAI 客户端
API_KEY = "sk-lolzalmconplprhxshsxunwdhpvlflbdbhneyyuyboqfbtsk"
URL = "https://api.siliconflow.cn/v1"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
client = AsyncOpenAI(api_key=API_KEY,base_url=URL)

# ---------- 定义API接口 ----------
# 当有POST请求到达"/chat"路径时，调用chat函数处理
@app.post("/chat")
async def chat(
        # Body(...) 表示这个参数必须从请求体中获取
        query: str = Body(..., description="用户输入"),
        sys_prompt: str = Body(
            "你是一个爱使用颜文字的私人助手",
            description="系统提示词，用于设定 AI 的行为"
        ),
        history: List[dict] = Body(
            [],
            description="历史对话记录，格式为 [{role: 'user/assistant', content: '消息内容'}]"),
        history_len: int = Body(5, description="上下文长度"),
        temperature: float = Body(0.5, description="温度"),
        top_p: float = Body(0.5, description="LLM采样概率"),
        max_tokens: int = Body(1024, description="最大token数"),
):
    """
    处理用户的聊天请求，并以流式传输的方式返回 AI 的回答。
    query: 用户当前的输入内容
    sys_prompt: 指导 AI 行为的系统指令
    history: 之前的对话历史，用于实现多轮对话的上下文关联
    history_len: 限制历史记录的长度，防止上下文过长导致模型性能下降或费用增加
    """
    # ---------- 构建消息列表 ----------
    # 每次请求都创建一个新的列表，解决全局变量的线程安全问题
    messages = []

    # 处理历史消息
    if history and history_len > 0:
        # 只保留最后 N 轮对话，每轮对话包含一个 user 和一个 assistant 的消息
        # history[-2 * history_len:] 可以精确地截取最近的 N 轮
        history = history[-2 * history_len:]

    # 添加系统提示词
    messages.append({"role": "system", "content": sys_prompt})
    # 添加历史消息
    messages.extend(history)
    # 添加用户输入
    messages.append({"role": "user", "content": query})

    # ---------- 调用大模型 ----------
    # 发送请求，await 表示等待异步操作完成
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=True
    )

    # ---------- 定义流式响应的生成器 ----------
    async def stream_response():
        """异步生成函数，逐个产生 AI 生成的文本片段"""
        async for chunk in response:  # 异步迭代响应
            chunk_msg = chunk.choices[0].delta.content
            if chunk_msg:
                yield chunk_msg  # 输出流式结果

    # ---------- 返回流式响应 ----------
    # StreamingResponse 是 FastAPI 提供的一个特殊响应类
    # 可以返回一个异步生成器，用于生成流式响应
    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",  # 推荐的 MIME 类型，便于前端处理
        headers={
            "Cache-Control": "no-cache",  # 告诉浏览器不要缓存响应
            "Connection": "keep-alive"  # 保持 HTTP 连接不关闭
        }
    )


if __name__ == '__main__':
    # 直接在主函数中启动服务
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",  # 监听所有网络接口，允许外部访问
        port=7000,
        log_level="info",  # 日志级别
    )
