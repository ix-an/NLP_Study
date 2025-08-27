from modelscope import AutoModelForCausalLM, AutoTokenizer


# model_name = "Qwen/Qwen3-0.6B"    # 模型名称
model_name = "./models"    # 使用下载好的模型路径

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
# chat模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# 准备模型输入
prompt = "请告诉我周杰伦专辑的发行顺序"
messages = [
    {"role": "user", "content": prompt}
]
# 生成输入
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True  # Switches between thinking and non-thinking modes. Default is True.
)
# 生成输入的 token ids
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,    # 关键词解包
    max_new_tokens=32768
)
# 无论输入/输出，都是token ids
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("推理内容:", thinking_content)
print("回答:", content)
