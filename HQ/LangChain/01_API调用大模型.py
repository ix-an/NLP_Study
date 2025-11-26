from langchain_openai import OpenAI  # 补全类模型
from langchain_openai import ChatOpenAI  # 对话类模型

API_KEY = "sk-lolzalmconplprhxshsxunwdhpvlflbdbhneyyuyboqfbtsk"
URL = "https://api.siliconflow.cn/v1"
MODEL = "Qwen/Qwen2.5-7B-Instruct"

chat_model = ChatOpenAI(
    openai_api_key=API_KEY,
    base_url=URL,
    model=MODEL,
)

llm_model = OpenAI(
    openai_api_key=API_KEY,
    base_url=URL,
    model=MODEL,
)

print(f"ChatModels：{chat_model.invoke("你好").content}")
print(f"LLMs：{llm_model.invoke("你好")}")
