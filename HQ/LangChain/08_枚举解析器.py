# 定义模型
from langchain_openai import ChatOpenAI
model = ChatOpenAI(
    model=r"Qwen/Qwen2.5-7B-Instruct",
    openai_api_key=r"sk-lolzalmconplprhxshsxunwdhpvlflbdbhneyyuyboqfbtsk",
    base_url=r"https://api.siliconflow.cn/v1"
)


# 定义枚举类
from enum import Enum
class Color(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"

from langchain.output_parsers import EnumOutputParser
from langchain_core.prompts import PromptTemplate

# 实例化枚举解析器，注意这里的enum参数要传入定义的枚举类
enum_parser = EnumOutputParser(enum=Color)
enum_parser_instructions = enum_parser.get_format_instructions()

# 提示词
prompt = PromptTemplate(
    template="请将{query}解析为{parser_instructions}，只返回枚举值。",
    input_variables=["query"],
    partial_variables={"parser_instructions": enum_parser_instructions}
)
final_prompt = prompt.format(query="天空是什么颜色？")

# 调用模型
response = model.invoke(final_prompt)
print(response.content)  # blue
# 应用枚举解析器
parsed_res = enum_parser.parse(response.content)
print(parsed_res)  # Color.BLUE
