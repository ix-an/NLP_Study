"""把知识库分割为知识块"""
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import json

# 定义每个知识块的最小长度阈值（字符数）
CHUNK_MIN_LENGTH = 200

# 初始化文档分割管道，使用本地的BERT分词模型
p = pipeline(
    task=Tasks.document_segmentation,           # 指定任务类型为文档分割
    model='C:\HuggingFace\Bert-Segment',        # 指定使用的模型路径
    model_revision='master'                     # 指定模型版本
)

knowledge_path = "./knowledge/knowledge.json"    # 本地知识库路径

# 读取并处理本地知识库内容
knowledge = []
with open(knowledge_path, "r", encoding="utf-8") as f:
    knowledge = json.load(f)    # 从JSON文件中加载知识库数据

    kn_chunks = []    # 缓存分割后知识块的列表

    # 遍历知识库中的每个文档
    for doc in knowledge:
        print("-" * 50)
        print(doc['metadata'])  # 打印文档元数据信息

        # 使用管道对文档内容进行语义分割
        result = p(
            documents=doc['content']  # 传入待分割的文档内容
        )

        # 将分割结果按行分割成多个文本块
        chunks = result[OutputKeys.TEXT].strip().split('\n')

        tmp = ''    # 临时变量用于累积文本，确保每个知识块达到最小长度要求

        # 遍历所有分割后的文本块
        for chunk in chunks:
            if len(chunk.strip()) == 0:
                continue    # 跳过空行或只包含空白字符的块

            # 累积当前文本块到临时变量中
            tmp += chunk

            # 当累积的文本长度超过最小长度阈值时，创建一个新的知识块
            if len(tmp) > CHUNK_MIN_LENGTH:
                kn_chunks.append({
                    'metadata': doc['metadata'],    # 元数据信息：可解释性
                    'content': tmp,
                })
                tmp = ''  # 重置临时变量

        # 处理剩余的文本（即使未达到最小长度也作为一个独立的知识块）
        if len(tmp) > 0:
            kn_chunks.append({
                'metadata': doc['metadata'],
                'content': tmp,
            })

    # 将分割后的知识块写入新的JSON文件
    with open("./knowledge/knowledge_chunks.json", "w", encoding="utf-8") as f:
        json.dump(kn_chunks, f, ensure_ascii=False, indent=4)
        print("写入成功")
