from dotenv import load_dotenv
import os
load_dotenv('../RAG.env')
langsmith_key = os.getenv("LANGSMITH_API_KEY")

import pandas as pd
from langsmith import Client


# 读取CSV文件
df = pd.read_csv('../data/黑悟空.csv')

# 将 dataFrame 转换为列表元组的形式
example_inputs = list(df.itertuples(index=False, name=None))
# print(example_inputs)

# 创建 LangSmith 客户端
client = Client(api_key=langsmith_key)
# 创建数据集
dataset = client.create_dataset(dataset_name="example-code")
for q, a in example_inputs:
    client.create_example(
        inputs={"Question": q},
        outputs={"Answer": a},
        dataset_id=dataset.id
    )