# -------------------- 导入依赖 --------------------
from langchain.chains import RetrievalQA  # 检索问答链
from langchain_community.vectorstores import FAISS  # FAISS向量数据库
from langchain_huggingface import HuggingFaceEmbeddings  # HuggingFace 嵌入模型
from langchain_community.document_loaders import TextLoader  # 文本加载器
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 递归字符文本分割器
from langchain_openai import ChatOpenAI  # OpenAI 聊天模型
from dotenv import load_dotenv  # 加载环境变量
import os  # 文件路径

load_dotenv('../RAG.env')  # 加载 .env 文件中的环境变量
openai_api_key = os.getenv("OPENAI_API_KEY")

# -------------------- 创建大模型 --------------------
model = ChatOpenAI(
    model=r'glm-4.5-flash',
    openai_api_key=openai_api_key,
    base_url=r'https://open.bigmodel.cn/api/paas/v4'
)

# -------------------- 词嵌入+向量数据库+检索器 --------------------
# 加载文本
loader = TextLoader("../[all叶]全明星摄像头不要乱扫啊.txt", encoding="utf-8")
docs = loader.load()  # 加载文件内容, 返回一个 Document 对象列表

# 创建文本分割器：200个字一组的切片，每组有20个字重叠
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = text_splitter.split_documents(docs)  # 应用分割器

# 初始化嵌入模型
embedding = HuggingFaceEmbeddings(model_name=r"C:\HuggingFace\bge-large-zh-v1.5")
# 构建 FAISS 向量数据库
vs = FAISS.from_documents(chunks, embedding)
# 创建检索器，用于从向量存储中检索与查询最相关的文档
retriever = vs.as_retriever()

# -------------------- 创建提示词 --------------------
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
# 系统消息模板
sys_msg = SystemMessagePromptTemplate.from_template(
    "请根据已知信息回答问题。 \n 已知信息：{context}"
)
# 用户消息模板
human_msg = HumanMessagePromptTemplate.from_template("{question}")
# 合并为完整的聊天提示模板
chat_prompt = ChatPromptTemplate.from_messages([sys_msg, human_msg])

# -------------------- 创建链 --------------------
qa = RetrievalQA.from_chain_type(
    llm=model,  # 模型
    chain_type="stuff",  # 链类型，"stuff" 表示将所有文档内容 stuffed 到提示中
    retriever=retriever,  # FAISS 检索器实例
    chain_type_kwargs={"prompt": chat_prompt},  # 字典，配置问答链的关键参数
)

# -------------------- 测试 --------------------
# 用户问题
user_input = "荣耀职业选手群第一个说话的是谁？"
# 检索器检索出最相关的文档
related_docs = retriever.invoke(user_input, k=2)
# 打印相关文档
print("检索器检索出的相关文档：")
for i, doc in enumerate(related_docs):
    print(f"文档{i+1}：\n {doc.page_content}")
    print("-" * 50)

# 链执行
result = qa.invoke(user_input)
print(f"问答链生成的答案：\n {result['result']}")