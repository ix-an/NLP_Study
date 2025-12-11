"""简化版 RAG + RAGAS 模板"""

# --------------------------------------
# 1. 必要的导入
# --------------------------------------
from dotenv import load_dotenv
import os
load_dotenv("RAGAS.env")
openai_api_key = os.getenv("OPENAI_API_KEY")
model_url = os.getenv("MODEL_URL")
bge_large_v15_path = os.getenv("BGE_MODEL_PATH")

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

# --------------------------------------
# 2. 构造一个“最小 RAG 系统”
# --------------------------------------

# 步骤 2.1: 手动构造知识库（最小 demo）
# 每个元素代表一个“文档 chunk”
docs = [
    "故宫位于北京，是明清两代的皇家宫殿。",
    "乔布斯是苹果公司联合创始人，2011年逝世。",
    "量子计算是一种基于量子比特的计算方式。"
]

# 步骤 2.2: 最简单的“检索器”
def simple_retriever(query, docs):
    keywords = ["故宫", "乔布斯", "量子计算"]  # 简化 tokenizer
    return [d for d in docs if any(k in d for k in keywords if k in query)]
    # 列表推导式：遍历 docs 中的每个文档 d，由 any(...) 判断是否保留
    # any(...) 逻辑：
    # 1. 遍历keywords 中的每个关键词 k：当前检索器的逻辑是关键词检索
    # 2. 如果k在query中，即：检索到query里有这个关键词k → 保留k
    # 3. 保留的k：如果当前d包含保留的这个K,证明这是相关上下文，应当保留d


# 步骤 2.3: 构造一个最简单的 RAG 生成器（LCEL 风格）
# 原因：我们需要“question + context → answer”
llm = ChatOpenAI(
    model="Qwen/Qwen2.5-7B-Instruct",
    base_url=model_url,
)

def rag_answer(question: str):
    """最小实现：检索相关文档 → 拼接 → 输入 LLM 生成答案"""
    contexts = simple_retriever(question, docs)

    # 如果没有检索到，就告诉模型不要乱说
    context_text = "\n".join(contexts) if contexts else "（无相关资料）"

    prompt = f"""
你是一个 RAG 系统，请严格基于以下 context 回答问题。
如果 context 中没有答案，请说“无法根据资料回答”。

【Context】
{context_text}

【Question】
{question}
"""
    answer = llm.invoke(prompt)
    return answer.content, contexts  # 返回 answer + 所用 contexts



# --------------------------------------
# 3. 构造用于 RAGAS 测试的数据（手写即可）
# --------------------------------------

# 注意：reference（真实答案）是必要的，
# 因为 Context Recall 需要它。
questions = [
    "故宫在哪里？",
    "乔布斯是谁？",
]

references = [
    "故宫位于北京。",
    "乔布斯是苹果公司联合创始人。",
]

# 用 RAG 生成 answers + contexts
answers = []
contexts = []

for q in questions:
    ans, ctx = rag_answer(q)
    answers.append(ans)
    contexts.append(ctx)


# --------------------------------------
# 4. 构造 RAGAS 输入数据集
# --------------------------------------

# 使用 HuggingFace 的 datasets 格式
dataset = Dataset.from_dict({
    "question": questions,  # 提问
    "answer": answers,  # RAG的答案
    "contexts": contexts,  # 检索到的上下文
    "reference": references  # 标准答案
})


# --------------------------------------
# 5. 调用 RAGAS 的四大指标
# --------------------------------------
metrics = [
    faithfulness,         # 答案是否忠实于 context？
    answer_relevancy,     # 答案是否真正回答了问题？
    context_precision,    # 检索的内容是否“纯净”？
    context_recall,       # 应该找的内容有没有漏？
]

# 创建embedding模型，用于语义相似度：answer_relevancy 必需
embeddings = HuggingFaceEmbeddings(model_name=bge_large_v15_path)

results = evaluate(
    dataset=dataset,
    metrics=metrics,
    llm=llm,               # RAGAS 评估用的 LLM
    embeddings=embeddings,  # RAGAS 评估用的 embedding 模型（语义相似度）
)


# --------------------------------------
# 6. 输出结果
# --------------------------------------
df_res = results.to_pandas()
print(df_res)

# 保存结果
df_res.to_csv("ragas_result.csv", index=False)
print("\n评估完成！结果已保存到 ragas_result.csv")