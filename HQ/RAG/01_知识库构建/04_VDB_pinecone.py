from dotenv import load_dotenv
import os
load_dotenv('VDB.env')
pinecone_api_key = os.getenv('PINECONE_API_KEY')

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# ----------------------------------------
# 📍创建Pinecone数据库：创建完成后，将数据库创建的代码注释掉
# ----------------------------------------
# 初始化Pinecone数据库（连接到Pinecone服务器）
pc = Pinecone(api_key=pinecone_api_key)
# 索引名称（数据库名字）
index_name = "pinecone-demo"
if not pc.has_index(index_name):  # 若索引不存在
    # 创建索引
    pc.create_index(
        name=index_name,
        dimension=1024,
        spec=ServerlessSpec(
            cloud="aws",  # 索引所在云（不用管，默认亚马逊）
            region="us-east-1",  # 索引所在区域（不用管，默认美国东部1）
        ),
        metric="cosine",  # 相似度计算方法
    )
    print(f"远程的数据库创建成功，库名为: {index_name}")


# ----------------------------------------
# 📍构建数据
# ----------------------------------------
# 加载Embedding模型
embed_model = SentenceTransformer(r'C:\HuggingFace\Qwen3-Embedding-0.6B')

# 设置输入文本
texts = ["猫猫喜欢吃鱼", "狗狗喜欢吃肉", "今天天气不错"]

# 向量化
embeds = embed_model.encode(texts)
# print(embeds.shape)  # (3, 1024)

# 创建查询向量：注意，向量化接收的是列表
query = embed_model.encode(["猫猫喜欢吃什么？"])

# ----------------------------------------
# 📍向索引添加数据（查询时需注释掉）
# ----------------------------------------
# 获取数据库索引对象
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("pinecone-demo")

# 向索引中插入向量
# index.upsert(
#     vectors=[
#         {"id": "vec1", "values": embeds[0].tolist()},
#         {"id": "vec2", "values": embeds[1].tolist()},
#         {"id": "vec3", "values": embeds[2].tolist()},
#     ]
# )
# 列表推导式
vectors = [
    {"id": f"vec{i}", "values": embeds[i].tolist()}
    for i in range(len(texts))
]
index.upsert(vectors=vectors)

# ----------------------------------------
# 📍相似度查询
# ----------------------------------------
# 获取数据库索引对象
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("pinecone-demo")

# 查询相似向量
res = index.query(
    vector = query[0].tolist(),  # 查询向量
    top_k=2,  # 返回最相似的K个向量
    include_values=False,  # 是否返回查询的实际向量
)

print(res)
"""
res ：查询结果，dict类型
- matches：查询结果列表，每个元素为一个dict，包含以下键值对：
    - id：向量ID
    - score：相似度得分
    - values：向量实际值（空列表因为include_values=False）
- namespace：命名空间（默认空字符串）
- usage：查询使用情况，嵌套dict：
    - read_units：读取单位数
"""

# ----------------------------------------
# 📍删除向量数据库
# ----------------------------------------
pc = Pinecone(api_key=pinecone_api_key)
pc.delete_index("pinecone-demo")