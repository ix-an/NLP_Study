from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.cluster import KMeans
import numpy as np

# ----------------------------------------
# 读取文本
# ----------------------------------------
loader = TextLoader('../data/[all叶]全明星摄像头不要乱扫啊.txt', encoding='utf-8')
docs = loader.load()
text = docs[0].page_content

# ----------------------------------------
# 按“句子”做初步切分
# ----------------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    separators=['\n\n', '\n', "。", "，"],  # 定义分隔符
    chunk_size=200,  # 块大小
    chunk_overlap=50  # 相邻块重叠字符
)
sentences = text_splitter.split_text(docs[0].page_content)

print(f"初步切成了 {len(sentences)}个句子。")
print(sentences[0])

# ----------------------------------------
# Embedding：对每个句子进行向量化
# ----------------------------------------
embed = HuggingFaceEmbeddings(
    model_name=r"C:\HuggingFace\Qwen3-Embedding-0.6B",
    model_kwargs={"device": "cuda"}
)

sentence_vectors = embed.embed_documents(sentences)
# 转为numpy 数组：计算快 + 适配 sklearn 库输入
sentence_vectors = np.array(sentence_vectors)

print("句向量 shape:", sentence_vectors.shape)

# ----------------------------------------
# 聚类：KMeans 切成 5 段
# ----------------------------------------
K = 5
kmeans = KMeans(n_clusters=K, random_state=42)
cluster_labels = kmeans.fit_predict(sentence_vectors)

print("聚类结果：", cluster_labels)

# ----------------------------------------
# 根据聚类结果合并为段落 + 排序
# ----------------------------------------
# 初始化空字典，key是聚类label（0-4），value是该label下的所有短句
clusters = {}
# 遍历每个短句的索引和对应的聚类label
for idx, label in enumerate(cluster_labels):
    # setdefault：如果label不在字典中，就创建key并赋值为空列表；如果存在，直接用现有列表
    clusters.setdefault(label, [])
    # 把当前短句（sentences[idx]）添加到对应label的列表中
    clusters[label].append(sentences[idx])

# 按label的数字从小到大排序
sorted_clusters = dict(sorted(clusters.items(), key=lambda x: x[0]))

# ----------------------------------------
# 输出语义段落
# ----------------------------------------
for label, sents in sorted_clusters.items():
    print(f"\n====== 语义段落 {label} ======")
    print("".join(sents))  # 把同一聚类的所有短句拼接成完整文本
    print("==============================")