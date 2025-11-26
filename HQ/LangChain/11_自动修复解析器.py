# 定义模型
from langchain_openai import ChatOpenAI
model = ChatOpenAI(
    model=r"Qwen/Qwen2.5-7B-Instruct",
    openai_api_key=r"sk-lolzalmconplprhxshsxunwdhpvlflbdbhneyyuyboqfbtsk",
    base_url=r"https://api.siliconflow.cn/v1"
)

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# 定义输出模型
class Person(BaseModel):
    name: str = Field(description="姓名")
    age: int = Field(description="年龄")

# JSON期望是双引号包围
mistake = "{'name': 'Alice', 'age': '18'}"

# 常使用Pyandtic解析器解析
parser_try = PydanticOutputParser(pydantic_object=Person)
try:
    output = parser_try.parse(mistake)
    print(output)
except Exception as e:
    print(e)  # Invalid json output

# 自动修复解析器
from langchain.output_parsers import OutputFixingParser
# 通过大模型尝试修复
parser_fix = OutputFixingParser.from_llm(parser=parser_try, llm=model)
fixed = parser_fix.parse(mistake)
print(f"fixed: {fixed}")
