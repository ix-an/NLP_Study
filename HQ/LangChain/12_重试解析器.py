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
class Action(BaseModel):
    action: str = Field(description="要采取的操作")
    action_input: str = Field(description="输入的操作")

# 定义Pydantic解析器
parser_pydantic = PydanticOutputParser(pydantic_object=Action)
parser_instructions = parser_pydantic.get_format_instructions()

# 提示词
prompt = PromptTemplate(
    template="请回答{query}，按照{parser_instructions}解析。",
    input_variables=["query"],
    partial_variables={"parser_instructions": parser_instructions}
)
# ⚠️⚠️⚠️ format() 返回的是str，需要用format_prompt()返回一个PromptValue对象
final_prompt = prompt.format_prompt(query="日剧《非自然死亡》开播时间？")

# 假设大模型给出错误回复：丢字段
mistake = '{"action":"search"}'

# ---------- 尝试使用自动修复解析器 -----------
from langchain.output_parsers import OutputFixingParser
parser_fix = OutputFixingParser.from_llm(parser=parser_pydantic, llm=model)
try:
    fixed = parser_fix.parse(mistake)
    print(f"fixed: {fixed}")
except Exception as e:
    print(f"fixed Error: {e}")

# ---------- 尝试使用重试解析器 -----------
from langchain.output_parsers import RetryWithErrorOutputParser
parser_retry = RetryWithErrorOutputParser.from_llm(parser=parser_pydantic, llm=model)
try:
    # ️⚠️️重试解析器（错误信息，提示词），且第二个参数必须是PromptValue 类型
    retry = parser_retry.parse_with_prompt(mistake, final_prompt)
    print(f"retry: {retry}")
except Exception as e:
    print(f"retry Error: {e}")
