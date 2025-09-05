import chromadb
from datetime import datetime

# 构建Embedding模型
from chromadb.utils import embedding_functions

chroma_client = chromadb.HttpClient(host="192.168.5.92", port=8081)

# 把文档信息转换为向量的模型
model_name = r"C:\HuggingFace\bge-reranker-base"
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=model_name,
)

if __name__ == "__main__":
    # 创建一个collection：类似在MySQL里面创建一个新的数据库
    collection = chroma_client.get_or_create_collection(
        name="my_collection",
        embedding_function=emb_fn,
        metadata={
            "description": "关于公司行政及出差、补贴等管理的知识库",
            "created": str(datetime.now()),
        },
    )
    # TODO：id应该是根据文档信息生成的md5值

    collection.add(
        ids=["1", "2", "3"],
        documents=["华清远见研发副总裁刘洪涛", "华清远见成立于2004", "今天天气很不错"],
        metadatas=[
            {"chapter": 3, "verse": 16},
            {"chapter": 3, "verse": 5},
            {"chapter": 29, "verse": 11}
        ],
    )

    # 打印记录数量
    print(collection.count())
    # 查看前2个记录
    print(collection.peek())
