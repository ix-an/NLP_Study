import gradio as gr
import requests

# FastAPI 后端地址
FASTAPI_URL = "http://localhost:7000/chat"


def chat_with_backend(prompt, history, sys_prompt, history_len, temperature, top_p, max_tokens, stream):
    # history: ["role": "user", metadata: {'title':None}, "content": "xxx"]
    # 去掉 metadata 字段
    history_new = [{"role": item["role"], "content": item["content"]} for item in history]

    # 构建请求数据
    data = {
        "query": prompt,
        "sys_prompt": sys_prompt,
        "history": history_new,
        "history_len": history_len,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens
    }

    # 发送请求到 FastAPI 后端
    response = requests.post(FASTAPI_URL, json=data, stream=True)

    if response.status_code == 200:
        full_response = ""
        if stream:
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                full_response += chunk
                yield full_response
        else:
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                full_response += chunk
            yield full_response


# 使用 gr.Blocks 创建一个块（构建界面），设置可以填充宽高
with gr.Blocks(fill_width=True, fill_height=True) as demo:
    # 创建一个标签页
    with gr.Tab("🦄ChatBot"):
        # 添加标题
        gr.Markdown("## 🦄 今天准备做什么？")

        # 创建一个行布局
        with gr.Row():
            # 创建一个左侧的列布局: 设置 AI 的参数（比例为1）
            with gr.Column(scale=1, variant="panel") as left_col:
                sys_prompt = gr.Textbox(label="系统提示词", value="你是一个爱使用颜文字的私人助手")
                history_len = gr.Slider(label="历史对话长度", minimum=0, maximum=10, value=5, step=1)
                temperature = gr.Slider(label="温度", minimum=0.01, maximum=2.0, value=0.7, step=0.01)
                top_p = gr.Slider(label="LLM采样概率", minimum=0.01, maximum=1.0, value=0.7, step=0.01)
                max_tokens = gr.Slider(label="最大token数", minimum=256, maximum=4096, value=1024, step=8)
                stream = gr.Checkbox(label="流式响应", value=True)

            # 创建一个列布局: 显示用户输入和AI的回答（比例为10）
            with gr.Column(scale=10) as main_col:
                # 创建一个Chatbot组件（聊天界面），高度为500px
                chatbot = gr.Chatbot(type="messages", height=500)
                # 创建ChatInterface，用于处理聊天的逻辑
                gr.ChatInterface(
                    fn=chat_with_backend,
                    chatbot=chatbot,
                    additional_inputs=[sys_prompt, history_len, temperature, top_p, max_tokens, stream],
                    type="messages"
                )

# 运行 Gradio 应用
if __name__ == "__main__":
    demo.launch(server_port=7860, inbrowser=True)