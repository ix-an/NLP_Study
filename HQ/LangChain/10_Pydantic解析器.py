# 定义模型
from langchain_openai import ChatOpenAI
model = ChatOpenAI(
    model=r"Qwen/Qwen2.5-7B-Instruct",
    openai_api_key=r"sk-lolzalmconplprhxshsxunwdhpvlflbdbhneyyuyboqfbtsk",
    base_url=r"https://api.siliconflow.cn/v1"
)

from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

# 定义输出模型
class EventDetails(BaseModel):
    event: str = Field(description="事件")
    date: str = Field(description="日期")

# 定义解析器
parser = PydanticOutputParser(pydantic_object=EventDetails)
parser_instructions = parser.get_format_instructions()

# 提示词
prompt = PromptTemplate(
    template="请回答{query}，按照{parser_instructions}解析。",
    input_variables=["query"],
    partial_variables={"parser_instructions": parser_instructions}
)
final_prompt = prompt.format(query="日剧《非自然死亡》开播时间？")

# 调用模型
response = model.invoke(final_prompt)
print(response.content)
# 应用解析器
print(parser.parse(response.content))
