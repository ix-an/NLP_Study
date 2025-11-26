"""内置的简单链 LLMChain
写法不够灵活，已被 LCEL逐渐取代
"""

# 定义模型
from langchain_openai import ChatOpenAI
model = ChatOpenAI(
    model=r"Qwen/Qwen2.5-7B-Instruct",
    openai_api_key=r"sk-lolzalmconplprhxshsxunwdhpvlflbdbhneyyuyboqfbtsk",
    base_url=r"https://api.siliconflow.cn/v1"
)

# 创建提示词
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个语气可爱的个人助手"),
    ("human", "{query}")
])

# 创建LLMChain
from langchain.chains import LLMChain
# 旧版语法: verbose=True 可打印显示详细信息
llm_chain = LLMChain(llm=model, prompt=prompt, verbose=True)
print(llm_chain("请写一篇200字的《小王子》读后感"))

# LCEL链
# 新版语法: 直接用 | 连接模型和提示词
llm_chain = prompt | model
response = llm_chain.invoke({"query": "请写一篇200字的《小王子》读后感"})
print(response.content)
