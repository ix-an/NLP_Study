# 定义模型
from langchain_openai import ChatOpenAI
model = ChatOpenAI(
    model=r"Qwen/Qwen2.5-7B-Instruct",
    openai_api_key=r"sk-lolzalmconplprhxshsxunwdhpvlflbdbhneyyuyboqfbtsk",
    base_url=r"https://api.siliconflow.cn/v1"
)

from langchain.output_parsers import StructuredOutputParser
from langchain.output_parsers import ResponseSchema  # 响应模式
from langchain.prompts import PromptTemplate

# 定义响应模式
response_schemas = [
    ResponseSchema(name="event", description="事件"),
    ResponseSchema(name="date", description="时间"),
]

# 定义结构化输出解析器，传入响应模式
structured_parser = StructuredOutputParser(response_schemas=response_schemas)
structured_parser_instructions = structured_parser.get_format_instructions()

# 提示词
prompt = PromptTemplate(
    template="请回答{query}，按照{parser_instructions}解析。",
    input_variables=["query"],
    partial_variables={"parser_instructions": structured_parser_instructions}
)
final_prompt = prompt.format(query="日剧《非自然死亡》开播时间？")

# 调用模型
response = model.invoke(final_prompt)
print(response.content)
# 应用结构化解析器
parsed_res = structured_parser.parse(response.content)
print(parsed_res)

