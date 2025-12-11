from dotenv import load_dotenv
import os
load_dotenv("./RAGAS.env")
openai_api_key = os.getenv("OPENAI_API_KEY")
model_url = os.getenv("MODEL_URL")
bge_large_v15_path = os.getenv("BGE_MODEL_PATH")

# --------------------------------------------------
# 创建模型对象：需要用LangchainLLMWrapper包装OpenAI模型
# --------------------------------------------------
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

chat_model = LangchainLLMWrapper(
    ChatOpenAI(
        model=r"Qwen/Qwen2.5-7B-Instruct",
        api_key=openai_api_key,
        base_url=model_url,
    )
)

# --------------------------------------------------
# 创建词嵌入模型，需要用LangchainEmbeddingsWrapper包装
# --------------------------------------------------
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

embedding_model = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name=bge_large_v15_path)
)


# --------------------------------------------------
# 计算上下文精度
# --------------------------------------------------
from ragas.metrics._context_precision import LLMContextPrecisionWithReference
from ragas import SingleTurnSample

cp = LLMContextPrecisionWithReference(llm=chat_model)

sample = SingleTurnSample(
    user_input="故宫位于哪里？",  # 用户提问
    response="故宫在北京。",  # LLM回答
    reference="故宫位于北京。",  # 真实答案
    retrieved_contexts=["北京是中国的首都。","故宫位于北京。"]  # 检索到的上下文
)

result = cp.single_turn_score(sample)
print(result)

"""
其它评估指标的查询类似，只是换了不同的API
"""