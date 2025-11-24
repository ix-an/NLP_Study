from transformers import AutoModelForCausalLM  # 因果语言模型加载器
from transformers import AutoTokenizer  # 自动分词
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model_dir = r"C:\HuggingFace\Qwen3-0.6B"  # 使用本地下载好的模型

# 加载模型，并移动到GPU
model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)
# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# 设置 prompt
prompt = "你好吗？我很好。"
# 获取输入
messages = [
    {"role": "user", "content": prompt},
    {"role": "system", "content": "你是一个爱用颜文字的个人助手。"}
]

# 按模型规则「格式化文本」，返回值是str
text = tokenizer.apply_chat_template(
    messages,                    # 输入
    tokenize=False,              # 是否分词
    add_generation_prompt=True,  # 是否添加生成提示
    enable_thinking=True         # 是否启用思考
)

# 「数字化文本」，切词并转换为模型能读取的 token ids（dict）
# 字典的键主要有：
# input_ids：(batch_size, sequence_length)，token ids 序列
# attention_mask：注意力掩码（0 表示忽略，1 表示关注）
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
# return_tensors: "pt"表示返回是PyTorch张量，可选："tf"、"np"

# 预测新 token序列
generated_ids = model.generate(
    **model_inputs,  # 关键词解包
    max_new_tokens=512
)


# 提取「仅新生成的token ids」（排除原始输入部分）
# 原始输入的 token 长度 = model_inputs["input_ids"] 的长度
input_length = model_inputs["input_ids"].shape[1]
# 从输入长度后开始截取，包含：思考内容 + 分隔标记 + 回答内容
new_generated_ids = generated_ids[0][input_length:]
output_ids = new_generated_ids.tolist()  # 转为列表方便处理

# 分割思考内容和回答内容
try:
    # 找到分隔标记（151668）在output_ids中的位置
    # output_ids[::-1]：将token序列反转（从后往前查，避免多个标记时出错）
    # len(output_ids) - 反转后的位置：换算成原序列中分隔标记的索引
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    # 如果没找到分隔标记（比如模型没生成推理内容），index设为0
    index = 0

# 解码 token ids 为文本
thinking_content = tokenizer.decode(
    output_ids[:index],  # 思考内容（分隔标记之前）
    skip_special_tokens=True
).strip("\n")

content = tokenizer.decode(
    output_ids[index:],  # 回答内容（分隔标记之后）
    skip_special_tokens=True
).strip("\n")


# 打印结果
print("=== 输入 ===")
print(prompt)
print("\n=== 推理内容 ===")
print(thinking_content if thinking_content else "无推理内容")
print("\n=== 回答内容 ===")
print(content)



