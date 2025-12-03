from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient

# 创建词向量文档和查询向量
model = SentenceTransformer(r'C:\HuggingFace\Qwen3-Embedding-0.6B')
texts = ["猫猫喜欢吃鱼", "狗狗喜欢吃肉", "今天天气不错"]
embeddings = model.encode(texts).tolist()
query = model.encode(["猫猫喜欢吃什么？"]).tolist()

"""
Milvus 第一次连接时默认进入 default 数据库
先连接才能创建新数据库，创建只需一次，之后每次连接直接指定 db_name
新 API（MilvusClient）会自动创建 schema 与索引
"""
# 连接Milvus服务器 -> 连接到"default"数据库
client = MilvusClient(uri="http://localhost:19530")
# 创建一个新的数据库："my_db" -> 创建后注释掉
# client.create_database("my_db")
# 连接到这个新数据库
# client = MilvusClient(
#     uri="http://localhost:19530",
#     db_name="my_db",
# )

# 创建collection
if not client.has_collection("demo_collection"):
    client.create_collection(
        collection_name="demo_collection",
        dimension=len(embeddings[0]),  # 自动匹配维度
        metric_type="COSINE",  # 可选："L2"、"COSINE"、"INNER_PRODUCT"
    )
# 加载collection到内存
client.load_collection("demo_collection")

# 插入数据
data = [
    {"id": i, "vector": embeddings[i], "text": texts[i]}
    for i in range(len(texts))
]

client.insert(
    collection_name="demo_collection",
    data=data,
)

# 刷新索引：因为Milvus是异步索引，需要刷新才能立即查询
client.flush("demo_collection")

# search前一定要再加载一次collection（幂等） -> "安全带"
client.load_collection("demo_collection")
# 搜索
response = client.search(
    collection_name="demo_collection",
    data=query,  # 查询向量
    limit=2,  # 返回Top2结果
    output_fields=["text"],  # 可选，返回原文
)

print(response)