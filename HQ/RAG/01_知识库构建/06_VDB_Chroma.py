from sentence_transformers import SentenceTransformer
from chromadb import Client, PersistentClient


# Embedding
model = SentenceTransformer(r'C:\HuggingFace\Qwen3-Embedding-0.6B')
texts = ["猫猫喜欢吃鱼", "今天天气不错", "狗狗喜欢吃肉"]
embeddings = model.encode(texts).tolist()

# ----------------------------------------
# 客户端创建
# ----------------------------------------
# 方式1：内存客户端 -> 运行结束数据丢失
# client = Client()
# 方式2：本地持久化客户端 -> 数据存在本地
client = PersistentClient(path="./db")

# ----------------------------------------
# 创建集合：类似于关系数据库中的「表」，是存储向量的容器
# ----------------------------------------
collection = client.get_or_create_collection(
    name="my_collection",
    metadata={"hnsw:space": "cosine"},
)

# ----------------------------------------
# 存储、检索
# ----------------------------------------
collection.add(
    ids=["id1", "id2", "id3"],  # 向量id，必填
    embeddings=embeddings,  # 向量，必填
    documents=texts,  # 关联原文，检索时能直接返回原文，可选
)

# 创建查询向量
query = model.encode(["猫猫喜欢吃什么？"]).tolist()
# 检索Top2相似结果
results = collection.query(
    query_embeddings=query,
    n_results=2,
    include=["documents", "distances"],  # 可选，返回原文和相似度/距离
)
print("\n检索结果：")
print(f"相似ID：{results['ids'][0]}")
print(f"相似原文：{results['documents'][0]}")
print(f"余弦距离：{results['distances'][0]}")
print("公式有所不同，这里的余弦距离为 1-cosine，越小越相似")

# ----------------------------------------
# 删除操作
# ----------------------------------------
# 删除指定ID的数据
collection.delete(ids=["id2"])
# 清空集合：先获取所有ID，再删除
collection.delete(ids=collection.get()["ids"])
# 删除集合
client.delete_collection(name="my_collection")