"""PromptTemplate"""
from langchain_core.prompts import PromptTemplate

# 定义模板
template = """
Hello，{name}！I {attitude} you.
"""

# PromptTemplate 类 从模板字符串创建一个 Prompt 对象
# ✡ from_template() 方法 直接从模板创建
prompt = PromptTemplate.from_template(template=template)
# ✡ 用类创建对象，并显示指定变量和模板
prompt = PromptTemplate(
    input_variables=["name", "attitude"],
    template=template
    )

# 使用方式一样，用 format_prompt() 方法格式化
# format() 方法：返回字符串，只能用于快捷格式化
# format_prompt() 方法：返回 PromptValue 对象，适用于后续高级处理！
print(prompt.format_prompt(name="Alice", attitude="like"))


print("\n ------------------------------------------ \n")


from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# 系统提示
system_text = "你是一个爱使用颜文字的个人助手"
system_template = SystemMessagePromptTemplate.from_template(system_text)

# 用户消息模板
human_text = "请将{text}翻译为{language}"
human_emplate = HumanMessagePromptTemplate.from_template(human_text)

# 整合提示模板
prompt_template = ChatPromptTemplate.from_messages([system_template, human_emplate])

# format_messages() 方法 格式化消息
prompt = prompt_template.format_messages(text="你好", language="英文")
print(prompt)
