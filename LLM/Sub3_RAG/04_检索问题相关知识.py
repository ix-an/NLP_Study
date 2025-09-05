"""检索和问题相关的知识块"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json

# 1. 加载本地知识块
chunks_path = "./knowledge/knowledge_chunks.json"
with open(chunks_path, "r", encoding="utf-8") as f:
    chunks_data = json.load(f)

documents = [c["content"] for c in chunks_data]

# 2. 加载 reranker 模型
model_name = "BAAI/bge-reranker-base"   # 可以换成 "BAAI/bge-reranker-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 3. 输入问题
query = "仙某某有哪些别称"

# 4. 计算相似度（relevance score）
scores = []
for idx, doc in enumerate(documents):
    # 输入 query + document
    inputs = tokenizer(
        query, doc,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        score = model(**inputs).logits.squeeze().item()

    scores.append((idx, doc, score))

# 5. 根据分数排序
scores = sorted(scores, key=lambda x: x[2], reverse=True)

# 6. 设置阈值，取前5个
threshold = 0.7
contexts, metadatas = [], []

for idx, doc, score in scores[:5]:
    if score >= threshold:
        contexts.append(doc)
        meta = chunks_data[idx].get("metedata", None)
        if meta and meta not in metadatas:
            metadatas.append(meta)

# 7. 打印结果
print("=== Rerank Top Results ===")
for c in contexts:
    print(">>", c)

print("\n=== Metadata ===")
print(metadatas)

