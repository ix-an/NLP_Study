from langsmith.schemas import Run, Example

def my_evaluator(run: Run, example: Example) -> dict:
    """自定义评估器：判断模型回答简洁性（是否少于2倍参考答案长度）"""
    prediction = run.outputs["output"]
    required = example.outputs["Answer"]
    score = len(prediction) < 2 * len(required)
    return {"key": "is_concise", "score": int(score)}

