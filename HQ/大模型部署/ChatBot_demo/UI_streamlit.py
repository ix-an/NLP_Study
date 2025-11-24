import streamlit as st  # 导入 Streamlit 库，用于创建 Web 应用
import requests  # 导入 requests 库，用于发送 HTTP 请求

# 启动命令：streamlit run UI_streamlit.py
# FastAPI 后端地址
FASTAPI_URL = "http://localhost:7000/chat"


# 设计页面
st.set_page_config(page_title="ChatBot", page_icon="🦄", layout="centered")

# 设计聊天对话框
st.title("🦄 今天有什么计划？")

# st.sidebar：设计侧边栏
with st.sidebar:  # 可以省略 st.sidebar.title 中的 sidebar
    st.title("ChatBot")
    sys_prompt = st.text_input("系统提示词", value="你一个爱使用颜文字的私人助手")
    # slider 做侧边栏滑块，允许用户动态调整
    # value: 滑块的初始值，step: 滑块的拖动步长
    history_len = st.slider("保留历史对话数量", min_value=0, max_value=10, value=5, step=1)
    temperature = st.slider("温度", min_value=0.01, max_value=2.0, value=0.7, step=0.01)
    top_p = st.slider("LLM采样概率", min_value=0.01, max_value=1.0, value=0.7, step=0.01)
    max_tokens = st.slider("最大token数", min_value=256, max_value=4096, value=1024, step=8)
    # checkbox 做侧边栏勾选框，允许用户选择是否开启流式响应
    stream = st.checkbox("流式响应", value=True)
    # button 按键
    st.button("清空聊天记录", on_click=lambda: st.session_state.history.clear())

# 定义存储历史
if "history" not in st.session_state:
    st.session_state.history = []

# 显示历史对话
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 输入框
# 海象运算符：如果 query 不为 None，则执行后面的代码块
if query := st.chat_input("询问任何问题"):
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(query)

    # 构建请求数据
    data = {
        "query": query,
        "sys_prompt": sys_prompt,
        "history": st.session_state.history,
        "history_len": history_len,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    # 发送请求到 FastAPI 后端
    response = requests.post(FASTAPI_URL, json=data, stream=stream)
    if response.status_code == 200:  # 响应成功
        # 创建一个空字符串，用于存储 AI 的回答
        chunks = ""

        # 创建一个占位符，用于显示 AI 的回答
        assistant_placeholder = st.chat_message("assistant")
        assistant_text = assistant_placeholder.markdown("")

        # 流式输出
        if stream:
            # 解析返回内容
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    # 追加内容
                    chunks += chunk
                    # 实时显示和更新 AI 的回答
                    assistant_text.markdown(chunks)
        else:
            assistant_text.markdown(response.text)

        # 存储到历史记录
        st.session_state.history.append({"role": "user", "content": query})
        st.session_state.history.append({"role": "assistant", "content": chunks})
