from fastapi import FastAPI

app = FastAPI()

# 运行：uvicorn 01_path_params:app --reload --port 7000

# chat_id 是路径参数，page和skip 是查询参数
@app.get("/query_chat/{chat_id}")
async def query_chat(
        chat_id: int,  # 必填参数
        page: int = 1,  # 可选参数
        skip: bool = False  # 布尔型自动转换（1/yes→True，0/no→False）
):
    return {
        "chat_id": chat_id,
        "page": page,
        "skip": skip
    }
