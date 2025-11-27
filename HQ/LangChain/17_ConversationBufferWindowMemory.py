# 定义模型
from langchain_openai import ChatOpenAI
model = ChatOpenAI(
    model=r"Qwen/Qwen2.5-7B-Instruct",
    openai_api_key=r"sk-lolzalmconplprhxshsxunwdhpvlflbdbhneyyuyboqfbtsk",
    base_url=r"https://api.siliconflow.cn/v1"
)

# 提示词
from langchain_core.prompts import PromptTemplate
template="""
你是一个与人类对话的机器人.
{chat_history}
Human: {human_input}
Chatbot:
"""
prompt=PromptTemplate(template=template, input_variables=["chat_history", "human_input"])

# 窗口缓存
from langchain.memory import ConversationBufferWindowMemory
memory = ConversationBufferWindowMemory(memory_key="chat_history", k=2)

# 定义链
from langchain.chains import LLMChain
chain = LLMChain(
    llm=model,
    prompt=prompt,
    verbose=True,  # 开启 verbose 模式，会打印出链的运行过程
    memory=memory
)

# 测试
print(chain.predict(human_input="你可以介绍一下成都吗?"))
print(chain.predict(human_input="你可以介绍一下北京吗?"))
print(chain.predict(human_input="你可以介绍一下上海吗?"))
print(chain.predict(human_input="我刚刚问过什么?请复述我的问题。"))