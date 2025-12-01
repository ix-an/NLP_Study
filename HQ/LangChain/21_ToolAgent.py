from dotenv import load_dotenv
load_dotenv()

import requests
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool  # 工具装饰器，用于定义工具函数
from langchain.agents import (
    create_tool_calling_agent,   # ⭐ 主流 Function Calling Agent
    AgentExecutor,
)


WEATHER_API_KEY = "Sa_TQtZ7nZ32nVtMd"


# -----------------------------------
# 1. 用 @tool 注册工具函数（会自动创建 function-calling schema））
# -----------------------------------
@tool(name_or_callable="天气查询", description="用于查询指定城市现在的天气和温度")
def fetch_weather(city: str) -> str:
    """查询指定城市的当前天气"""
    city = city.strip().split("\n")[0]  # 清除多余换行符和空格
    if not city:
        return "请提供有效的城市名称"

    url = (
        f"https://api.seniverse.com/v3/weather/now.json?"
        f"key={WEATHER_API_KEY}&location={city}&language=zh-Hans&unit=c"
    )

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        return f"请求失败: {str(e)}"

    try:
        data = response.json()
        if data.get("results"):
            weather = data["results"][0]["now"]
            return f"{city}当前天气：{weather['text']}，温度：{weather['temperature']}℃"
        else:
            return f"无法获取{city}的天气信息"
    except Exception as e:
        return f"数据解析错误: {str(e)}"


# -----------------------------------
# 2. LangChain 工具封装
# -----------------------------------
tools = [fetch_weather]


# -----------------------------------
# 3. 创建模型（支持 Function Calling 的模型都可以）
# -----------------------------------
chat_model = ChatOpenAI(
    model='glm-4.5-flash',
    base_url='https://open.bigmodel.cn/api/paas/v4',
    temperature=0.1,
)

# -----------------------------------
# 4. 创建提示词（简单清晰支持多轮，无需写大段思考模式）
# -----------------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个智能助手，在需要的时候会自动调用提供的工具。
    核心规则：
    1、思考是否需要调用工具，如果需要，必须调用，禁止编造答案。
    2、如果需要调用工具，必须严格按照格式要求。
    3、根据工具返回结果，整理最终回答。
    4、如果工具调用失败，如实告知用户错误原因。
    """),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")  # 模型工具调用历史，LangChain自动管理
])

# -----------------------------------
# 5. 创建 Function Calling Agent（最推荐）
# -----------------------------------
agent = create_tool_calling_agent(
    llm=chat_model,
    tools=tools,
    prompt=prompt,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,  # 处理解析错误
    max_iterations=3,  # 限制最大迭代次数，防止无限循环
)


# -----------------------------------
# 6. 测试
# -----------------------------------
if __name__ == "__main__":
    query = "重庆现在什么天气？"
    result = agent_executor.invoke({"input": query})
    print(result["output"])
