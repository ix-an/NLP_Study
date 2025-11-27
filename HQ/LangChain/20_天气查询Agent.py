import requests
from pydantic import Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.agents import (
    Tool,  # 封装工具
    create_react_agent,  # 创建ReAct代理
    AgentExecutor,  # 执行代理
)


# 相关配置 - 实际使用时移到环境变量或配置文件
API_KEY = "Sa_TQtZ7nZ32nVtMd"
MODEL = r"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MODEL_KEY = r"sk-lolzalmconplprhxshsxunwdhpvlflbdbhneyyuyboqfbtsk"
URL = r"https://api.siliconflow.cn/v1"


class WeatherTool:
    """天气查询工具类，用于获取指定城市的当前天气信息"""
    city: str = Field(description="城市名称")

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def run(self, city: str) -> str:
        """
        查询指定城市的当前天气

        Args:
            city: 城市名称

        Returns:
            包含天气信息的字符串，查询失败则返回错误信息
        """
        # 清除多余换行符和空格
        city = city.strip().split("\n")[0]
        if not city:
            return "请提供有效的城市名称"

        # 构建API请求URL
        url = (
            f"https://api.seniverse.com/v3/weather/now.json?"
            f"key={self.api_key}&"
            f"location={city}&"
            f"language=zh-Hans&"
            f"unit=c"
        )  # 括号自动拼接字符串

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # 抛出HTTP错误状态码
        except requests.exceptions.RequestException as e:
            return f"请求失败: {str(e)}"

        # 解析JSON响应
        try:
            data = response.json()
            if data.get("results"):
                weather = data["results"][0]["now"]
                return f"{city}的当前天气是：{weather['text']}，温度是：{weather['temperature']}℃"
            else:
                return f"无法获取{city}的天气信息"
        except (KeyError, ValueError) as e:
            return f"数据解析错误: {str(e)}"


# 创建天气工具实例
weather_tool = WeatherTool(API_KEY)

# 创建模型
chat_model = ChatOpenAI(
    model=MODEL,
    openai_api_key=MODEL_KEY,
    base_url=URL,
    temperature=0  # 降低随机性，提高输出稳定性
)

# 封装工具
tools = [Tool(
    name="天气查询",
    func=weather_tool.run,
    description="用于查询指定城市的当前天气和温度，输入应为城市名称（如：北京）",
)]

# 提示词模板
template = """请回答用户的问题。如果需要查询天气，请使用提供的工具。

可用工具:
{tools}

使用工具的格式要求非常严格，必须按照以下格式:
Thought: 思考是否需要使用工具，以及使用哪个工具
Action: [{tool_names}]
Action Input: 工具的输入参数（城市名称）

得到工具返回结果后，按以下格式:
Observation: 工具返回的结果

最后，整理结果给出最终回答:
Final Answer: 你的最终回答

开始处理:
Question: {input}
{agent_scratchpad}
"""
# 创建提示词
prompt = PromptTemplate.from_template(template=template)

# 创建代理
agent = create_react_agent(
    llm=chat_model,
    tools=tools,
    prompt=prompt,
    # 停止序列，避免输出混乱或超出步骤的多余信息
    stop_sequence=["\nObservation", "\nFinal Answer"]
)

# 创建代理执行器
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,  # 处理解析错误
    max_iterations=3  # 限制最大迭代次数，防止无限循环
)

# 测试
if __name__ == "__main__":
    query = "重庆现在什么天气？"
    result = agent_executor.invoke({"input": query})
    print(result["output"])