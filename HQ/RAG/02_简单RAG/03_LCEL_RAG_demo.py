import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough  # 透传

# --------------------------------------------------
# 基础配置与模型初始化
# --------------------------------------------------
# 加载环境变量
load_dotenv('../RAG.env')
openai_api_key = os.getenv("OPENAI_API_KEY")
model_url = os.getenv("MODEL_URL")
bge_large_v15_path = os.getenv("BGE_MODEL_PATH")
# 文件路径
file_path = '../[all叶]全明星摄像头不要乱扫啊.txt'
# 初始化 Chat 模型
chat_model = ChatOpenAI(
    model=r'glm-4.5-flash',
    openai_api_key=openai_api_key,
    base_url=model_url
)
# 初始化 Embedding 模型
embed_model = HuggingFaceEmbeddings(model_name=bge_large_v15_path)

# --------------------------------------------------
# 文档加载与切分
# --------------------------------------------------
def load_and_split_text(path, size=200, overlap=20):
    """加载并切分文本文件为多个文本块"""
    # 加载文本
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    # 初始化文本切分器
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '。', ','],
        chunk_size=size,
        chunk_overlap=overlap
    )
    # 切分文本
    result_chunks = text_splitter.split_text(text)
    return result_chunks

chunks = load_and_split_text(file_path)
print(f"文本块数量：{len(chunks)}")

# --------------------------------------------------
# 构建 FAISS 向量数据库，并创建检索器
# --------------------------------------------------
vs = FAISS.from_texts(
    texts=chunks,
    embedding=embed_model,
    metadatas=[
        {"chunk_id": i}
        for i in range(len(chunks))
    ]
)

retriever = vs.as_retriever(
    search_type="similarity",  
    search_kwargs={"k": 3}
)

# --------------------------------------------------
# Prompt 模板（LCEL会自动把变量注入）
# --------------------------------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个阅读助手，请根据提供的参考内容回答问题。"),
    ("user", "问题：{query}\n\n参考内容：\n{context}")
])

# --------------------------------------------------
# 文档合并函数：将检索结果合并为上下文字符串（非常重要）
# --------------------------------------------------
def merge_docs(docs):
    """
    输入：retriever 返回的 Document 列表
    输出：拼接后的字符串 context
    """
    return "\n".join([d.page_content for d in docs])

# --------------------------------------------------
# 输出解析器：解析ChatModel的输出
# --------------------------------------------------
out_parser = StrOutputParser()

# --------------------------------------------------
# LCEL RAG 链
# 输入：str，用户提问query
# 输出：str，模型最终的回答
# --------------------------------------------------

rag_chain = (
    # 输入路由器
    {
        "context": retriever | merge_docs,
        "query": RunnablePassthrough()
    }
    | prompt
    | chat_model
    | out_parser
)

# --------------------------------------------------
# 调用 LCEL RAG
# --------------------------------------------------
if __name__ == "__main__":
    query = "荣耀职业选手群第一个说话的是谁？"
    answer = rag_chain.invoke(query)
    print(answer)
