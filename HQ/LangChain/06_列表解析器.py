from langchain_core.prompts import ChatPromptTemplate
# 逗号分隔的列表输出解析器
from langchain.output_parsers import CommaSeparatedListOutputParser

# 定义模型
from langchain_openai import ChatOpenAI
model = ChatOpenAI(
    model=r"Qwen/Qwen2.5-7B-Instruct",
    openai_api_key=r"sk-lolzalmconplprhxshsxunwdhpvlflbdbhneyyuyboqfbtsk",
    base_url=r"https://api.siliconflow.cn/v1"
)


# 实例化输出解析器
parser = CommaSeparatedListOutputParser()
# 获取列表解析器的格式指令
paser_instructions = parser.get_format_instructions()
# print(paser_instructions)
# Your response should be a list of comma separated values, eg: `foo, bar, baz` or `foo,bar,baz`

# 定义提示词模板
prompt = ChatPromptTemplate.from_messages([
    # 提示 + 约束：让模型返回『符合解析器要求格式的文本』
    ("system", "{parser_instructions}"),  # 在Prompt中添加解析器提示词
    ("human", "列出5个{subject}色系的十六进制颜色码")
])

# 生成最终提示词
final_prompt = prompt.format(
    subject="莫兰迪",
    parser_instructions=paser_instructions
)
# print(final_prompt)

# 调用模型：模型只会返回文本字符串
response = model.invoke(final_prompt)
print(response.content)  # #CFB5CA,#847A8A,#B79D7B,#9A8FAB,#ADA08B

# 应用输出解析器，得到我们想要的列表格式
parsed_res = parser.parse(response.content)
print(parsed_res)  # ['#CFB5CA', '#847A8A', '#B79D7B', '#9A8FAB', '#ADA08B']



