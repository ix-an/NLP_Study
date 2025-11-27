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

# 摘要记忆
from langchain.memory import ConversationSummaryMemory
memory = ConversationSummaryMemory(llm=model, memory_key="chat_history")

# 定义链
from langchain.chains import LLMChain
chain = LLMChain(
    llm=model,
    prompt=prompt,
    verbose=True,  # 开启 verbose 模式，会打印出链的运行过程
    memory=memory
)

# 测试
print(chain.predict(human_input="I am Alice"))
print(chain.predict(human_input="Who am I?"))