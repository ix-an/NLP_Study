import os
os.environ['HF_HOME'] = "C:/HuggingFace/chinanews-chinese"

from transformers import *
import torch

# 1.初始化Tokenizer
tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-chinanews-chinese")

# 2.初始化Model
model = AutoModelForSequenceClassification.from_pretrained("uer/roberta-base-finetuned-chinanews-chinese")

# 3. 数据预处理
input_text = "计算机的春天在《微微一笑很倾城》"
inputs = tokenizer(input_text, return_tensors="pt")
# 返回是一个dict：包括 inputs_ids, token_type_ids, attention_mask
print("数据预处理后的输入:\n", inputs)

# 4.模型预测
res = model(**inputs)

# 5. 后处理
print("模型输出:\n", res)
logits = res.logits
probs = torch.softmax(logits, dim=-1)  # softmax归一化概率
print("概率:\n", probs)
# 将预测结果转换为类别标签
pred = probs.argmax(dim=-1).item()
result = model.config.id2label.get(pred)
print("预测结果:", result)
