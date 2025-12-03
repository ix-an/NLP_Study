from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embed_model = HuggingFaceEmbeddings(model_name=r"C:\HuggingFace\Qwen3-Embedding-0.6B")

texts = ['卧室里有一只可爱的小白猫','沙发上有一只酷酷的小黑猫','今天的天空依旧蓝蓝的']

# 构建FAISS向量存储（向量索引）
vs = FAISS.from_texts(
    texts=texts,
    embedding=embed_model,
    metadatas=[{"source": f"doc_{i}"} for i in range(len(texts))]
)

# 配置检索器
retriever = vs.as_retriever(
    search_type="similarity",  # 检索方式：相似度检索
    search_kwargs={"k": 2},  # 配置检索参数，k即TopK
)

# 检索
query = "白色的猫在哪里？"
res = retriever.invoke(query)

# 打印结果
for doc in res:
    print(f"文档来源：{doc.metadata['source']}")
    print(doc.page_content)