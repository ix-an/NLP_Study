from langchain.llms.base import LLM  # 续写类模型
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = r"C:\HuggingFace\Qwen3-0.6B"

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)

# 实现自定义LLM类
class QwenLLM(LLM):
    def _call(self, prompt: str, stop: str = None) -> str:
        # 将输入转换为模型输入
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        # 生成输出
        outputs = model.generate(**inputs, max_length=128)
        # 解码输出 -> 文本（str）
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output_text

    @property  # 装饰器：将方法变成属性
    def _identifying_params(self) -> dict:
        # 返回模型参数
        return {"model": model.config.name_or_path}

    @property
    def _llm_type(self):
        # 返回模型类型
        return "custom"


if __name__ == "__main__":
    local_llm = QwenLLM()
    result = local_llm("你好吗？我很好。")
    print(result)

