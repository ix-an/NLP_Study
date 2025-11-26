from langchain.llms.base import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = r"C:\HuggingFace\Qwen3-0.6B"
# 系统提示词
with open("./redbook_prompt.md", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)

# 实现自定义LLM类
class ChatQwenLLM(LLM):
    def _call(self, prompt: str, stop: str = None) -> str:
        # 消息封装
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        # 将消息列表转换为格式化的文本
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        """
        <|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        你好<|im_end|>
        <|im_start|>assistant
        """
        # 将输入的prompt转换成模型输入格式
        inputs = tokenizer([text], return_tensors="pt").to(device)
        # 生成输出
        outputs = model.generate(**inputs, max_new_tokens=2048, temperature=0.8)
        # 解码输出:outputs[0]为模型生成的文本 skip_special_tokens=True为跳过特殊字符
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    @property  # 装饰器：将方法变成属性
    def _identifying_params(self) -> dict:
        # 返回模型参数
        return {"model": model.config.name_or_path}

    @property
    def _llm_type(self):
        # 返回模型类型
        return "custom"

if __name__ == '__main__':
    local_llm = ChatQwenLLM()
    result = local_llm("请生成一个日常PLOG文案，代入软广")
    print(result)
