from fastapi import FastAPI
from pydantic import BaseModel

# 1. 实例化 FastAPI 应用（核心入口，所有接口都基于这个 app 定义）
app = FastAPI(
    title="请求体用法示例",  # 接口文档标题（访问 /docs 可看到）
    description="通过 Pydantic 模型定义请求体，处理复杂数据提交",  # 文档描述
    version="1.0"  # 版本号
)

# 2. 定义 Pydantic 模型（请求体的「数据结构模板」）
# 作用：规定客户端提交的 JSON 数据必须包含哪些字段、字段类型是什么
# FastAPI 会自动根据这个模型做 数据校验 + 类型转换 + 生成接口文档
class Item(BaseModel):
    # 必选字段：只写类型注解（str），无默认值 → 客户端必须传 name
    name: str
    # 可选字段：类型注解 + 默认值（None） → 客户端可传可不传，不传则为 None
    description: str | None = None  # str | None 等价于 Optional[str]（Python 3.10+ 写法）
    # 必选字段：类型为 float → 客户端必须传数字（整数/小数都会自动转 float）
    width: float
    # 必选字段：类型为 float → 同上
    height: float

# 3. POST 接口：创建新 Item（用请求体接收复杂数据）
@app.post(
    "/items/",  # 接口路径
    summary="创建一个新的物品",  # 接口文档中的简要说明
    description="接收物品的名称、描述、宽高，返回创建的物品信息"  # 详细描述
)
async def create_item(
    # 函数参数：item: Item → 告诉 FastAPI：这个接口需要接收一个符合 Item 模型的请求体
    # FastAPI 会自动做以下操作：
    # ① 解析客户端提交的 JSON 数据
    # ② 校验数据是否符合 Item 模型（比如少传 name 会报错）
    # ③ 把校验后的 JSON 转成 Python 对象（item.name、item.width 可直接访问）
    item: Item
):
    # 业务逻辑：这里可以对接数据库（比如把 item 存入数据库），这里简化为直接返回
    # 返回 item 对象时，FastAPI 会自动把它转成 JSON 格式返回给客户端
    return item

# 4. PUT 接口：更新已有 Item（路径参数 + 请求体结合使用）
@app.put(
    "/items/{item_id}",  # 接口路径：{item_id} 是路径参数（标识要更新的物品ID）
    summary="更新一个已有的物品",
    description="通过 item_id 找到物品，用请求体中的新数据更新它"
)
async def update_item(
    # 第一个参数：item_id: int → 路径参数（必须是整数，FastAPI 自动校验）
    item_id: int,
    # 第二个参数：item: Item → 请求体（包含更新后的物品数据）
    item: Item
):
    # 业务逻辑：这里可以根据 item_id 从数据库查询物品，再用 item 的数据更新
    # 简化处理：把 item 转成字典，和 item_id 合并后返回
    return {
        "item_id": item_id,  # 路径参数（要更新的物品ID）
        **item.dict()  # 把 Item 对象转成字典，展开合并（等价于 name=item.name, width=item.width...）
    }

# 运行命令：uvicorn 02_request_body:app --reload --port 7000
# 接口文档：http://127.0.0.1:7000/docs（可直接在网页上测试接口）

