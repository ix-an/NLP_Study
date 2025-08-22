"""-------创建Pipeline-------"""
import os
# 在代码开头设置环境变量：必须在导入任何 transformers 模块之前
os.environ['HF_HOME'] = "C:/HuggingFace/chinanews-chinese"

from transformers import pipeline


# # 1)直接使用任务类型创建，会下载一个默认的模型
# pipe = pipeline("text-classification")

# # 2)使用任务类型和模型名称创建，也会现场下载模型
# pipe = pipeline(
#     "text-classification",
#     model = "uer/roberta-base-finetuned-chinanews-chinese",
#     cache_dir="./models/chinanews-chinese",  # 指定缓存目录
# )



# # 然后正常创建 pipeline，不指定 cache_dir
# pipe = pipeline(
#     "text-classification",
#     model="uer/roberta-base-finetuned-chinanews-chinese"
# )


# 3)预先加载模型，再创建Pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model = AutoModelForSequenceClassification.from_pretrained(
    "uer/roberta-base-finetuned-chinanews-chinese",
)
tokenizer = AutoTokenizer.from_pretrained(
    "uer/roberta-base-finetuned-chinanews-chinese",
)
pipe = pipeline(
    "text-classification",
    model = model,
    tokenizer = tokenizer,
    device = 0,    # 指定使用第0块GPU
)

print(pipe("我喜欢叶修是因为叶修天下第一好！"))
