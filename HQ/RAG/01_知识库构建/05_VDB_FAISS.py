import faiss
from sentence_transformers import SentenceTransformer
import numpy as np  # FAISS要求输入的向量是numpy数组

# 创建词向量文档和查询向量
embed_model = SentenceTransformer(r'C:\HuggingFace\Qwen3-Embedding-0.6B')
docs_embeds = embed_model.encode(["猫猫喜欢吃鱼", "狗狗喜欢吃肉", "今天天气不错"])
query_embeds = embed_model.encode(["猫猫喜欢吃什么？"])
# 转为numpy数组
docs_embeds = np.array(docs_embeds)
query_embeds = np.array(query_embeds)

# 📍初始化faiss索引(数据库) ------------------------------
# 方式1：内积检索（标准化后等于余弦相似度）
index = faiss.IndexFlatIP(1024)
# 方式2：L2距离检索
# index = faiss.IndexFlatL2(1024)
# 添加索引
index.add(docs_embeds)
# 保存索引到本地磁盘：创建 "./db" 目录
faiss.write_index(index, "./db/faiss_index.faiss")

# 📍查询相似度 ------------------------------
faiss_index = faiss.read_index("./db/faiss_index.faiss")
D, I = faiss_index.search(query_embeds, k=2)
print(f"最相似的句子索引：{I}")
print(f"相似度分数（相似度/距离）：{D}")
