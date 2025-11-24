from fastapi import FastAPI

app = FastAPI()  # 创建FastAPI实例
# 运行：uvicorn first_FastAPI_web:app --reload --port 7000


# app.get()装饰器：将函数定义为HTTP GET请求的处理函数
@app.get("/")  # 定义根路由
async def read_root():  # 协程：异步处理请求，适合大模型调用这种IO密集型任务
    return {"msg": "大模型对话API启动成功！"}

# 带参数的GET接口（传递对话ID等）
@app.get("/chat/{chat_id}")
async def get_chat(chat_id: int, user_id: str = None):
    return {"chat_id": chat_id, "user_id": user_id,}



