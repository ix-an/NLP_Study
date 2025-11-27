from langchain_openai import ChatOpenAI
model = ChatOpenAI(
    model=r"Qwen/Qwen2.5-7B-Instruct",
    openai_api_key=r"sk-lolzalmconplprhxshsxunwdhpvlflbdbhneyyuyboqfbtsk",
    base_url=r"https://api.siliconflow.cn/v1"
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser



prompt = ChatPromptTemplate.from_template("说一句包含 {input} 的诗句")
parser = StrOutputParser()
# 定义链
chain = prompt | model | parser
print(chain.invoke({"input": "花"}))