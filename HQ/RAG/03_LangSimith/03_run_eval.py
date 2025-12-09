from dotenv import load_dotenv
import os
from langsmith import evaluate
from langsmith.schemas import Run, Example
from langsmith.evaluation import LangChainStringEvaluator
from langchain_openai import ChatOpenAI

# ----------------------------------------
# 配置环境变量
# ----------------------------------------
load_dotenv('../RAG.env')
openai_api_key = os.getenv('OPENAI_API_KEY')
openai_url = os.getenv('MODEL_URL')
langsmith_api_key = os.getenv('LANGSMITH_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# 裁判模型
judge_llm = ChatOpenAI(
    model="glm-4.5-flash",
    openai_api_key=openai_api_key,
    base_url=openai_url,
    temperature=0
)

# ----------------------------------------
# 定义 target_function
# LangSmith 会向它传 {"Question": "..."} 这样的字典
# ----------------------------------------
# 定义简单的rag
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
prompt = ChatPromptTemplate.from_template("请回答：{question}")
rag_chain = prompt | judge_llm | StrOutputParser()

# 定义 target_function
def target_fn(example_inputs: dict):
    q = example_inputs["Question"]
    result = rag_chain.invoke(q)
    # getattr(对象, 要获取的属性名, 默认值)
    answer_text = getattr(result, "content", str(result))
    return {"output": answer_text}


# ----------------------------------------
# Evaluator #1：规则评估
# 类似向量相似度，都是写代码，通过代码逻辑判断
# ----------------------------------------
def concise_evaluator(root_run: Run, example: Example):
    """
        root_run.outputs["output"]   : 模型输出
        example.outputs["Answer"]    : 参考答案
    """
    stu_answer = root_run.outputs.get("output", "")
    true_answer = example.outputs.get("Answer", "")

    # 简单规则：生成答案长度 < 参考答案两倍 → 认为简洁
    score = 1 if len(stu_answer) < len(true_answer) * 2 else 0

    return {
        "key": "is_concise",
        "score": score
    }


# ----------------------------------------
# Evaluator #2：使用 LangChain 内置的 QA 评估器（StringEvaluator）
# 它会自动判断 “模型回答是否符合参考答案”
# ----------------------------------------
qa_evaluator = LangChainStringEvaluator(
    evaluator="qa",
    config={"llm": judge_llm},
    prepare_data = lambda run, example: {
        "prediction": run.outputs.get("output"),  # 对应 RAG 的输出
        "reference": example.outputs.get("Answer"),  # 对应数据集的答案
        "input": example.inputs.get("Question"),     # 对应数据集的问题
    }
)


# ----------------------------------------
# Evaluator #3：自定义 LLM-as-a-Judge（可写提示词）
# ----------------------------------------
JUDGE_PROMPT = """
你是一个严谨的评估助理，请用 0 或 1 判断模型回答是否正确。

问题：{question}
参考答案：{reference}
模型回答：{output}

如果模型答案语义上正确 → 1
否则 → 0

请直接返回一个数字（0 或 1），不要解释。
"""

def llm_judge_evaluator(root_run: Run, example: Example):
    q = example.inputs["Question"]
    true_answer = example.outputs["Answer"]
    stu_answer = root_run.outputs["output"]

    _prompt = JUDGE_PROMPT.format(
        question=q,
        reference=true_answer,
        output=stu_answer
    )
    # 调用大模型打分
    result = judge_llm.invoke(_prompt)

    try:
        score = int(result.content.strip()[0])
    except ValueError:
        score = 0

    return {"key": "judge_correctness", "score": score}


# ----------------------------------------
# 主入口：调用 evaluate()
# ----------------------------------------
if __name__ == "__main__":
    print("🚀 开始评估，请稍等……")


    evaluate(
        target_fn,                            # 目前版本必须用位置参数
        data="example-code",                  # 在 LangSmith 创建的数据集名称
        evaluators=[
            concise_evaluator,                # 规则 evaluator
            qa_evaluator.as_run_evaluator(),  # 内置 QA 评估器
            llm_judge_evaluator               # LLM-as-a-Judge
        ],
        experiment_prefix="rag-eval-demo",    # 实验前缀，会在 LangSmith 里看到
        max_concurrency = 1                   # 最大并发数
    )

    print("🎉 完成！请打开 LangSmith dashboard 查看可视化结果！")