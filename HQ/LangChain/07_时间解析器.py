# 定义模型
from langchain_openai import ChatOpenAI
model = ChatOpenAI(
    model=r"Qwen/Qwen2.5-7B-Instruct",
    openai_api_key=r"sk-lolzalmconplprhxshsxunwdhpvlflbdbhneyyuyboqfbtsk",
    base_url=r"https://api.siliconflow.cn/v1"
)

from langchain.output_parsers import DatetimeOutputParser
from langchain_core.prompts import PromptTemplate


# 定义时间戳解析器并获取格式指令
time_parser = DatetimeOutputParser()
time_parser_instructions = time_parser.get_format_instructions()

# 创建提示词模板
prompt = PromptTemplate(
    template="请回答{query}，{parser_instructions}",
    input_variables=["query"],  # 只声明了query
    # partial_variables 可以提前为模板中的一个或多个变量填充固定的值
    # 这里提前绑定解析器指令，避免每次调用时都重复传递
    partial_variables={"parser_instructions": time_parser_instructions}
)

# 生成最终提示词
final_prompt = prompt.format(
    query="日剧《非自然死亡》开播时间？"
)

# 调用模型
response = model.invoke(final_prompt)
print(response.content)  # 2017-10-05T21:00:00.000000Z
# 应用时间戳解析器
parsed_res = time_parser.parse(response.content)
print(parsed_res)  # 2017-10-05 21:00:00

