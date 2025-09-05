import re
import json
from langchain_community.document_loaders import DirectoryLoader
# json文件不能通过DirectoryLoader加载


if __name__ == "__main__":
    loader = DirectoryLoader(
        "./data/",
        loader_kwargs={
            "languages" : ["chi_sim", "eng"],  # OCR时用中文和英文
        }
    )

    documents = loader.load()
    # print(documents)
    i = 0
    knowledge = []  # 存储知识
    for doc in documents:
        i += 1
        print("-" * 50)
        # print(f'第{i}份文档：{doc.metadata["source"]}, \n内容：{doc.page_content[:50]}')

        # 获取元消息的消息来源
        source = doc.metadata["source"]
        content = doc.page_content.replace(" ", "").replace("\n", " ")
        # 把多个空格替换为一个空格，使用正则表达式
        content = re.sub(r"\s+", " ", content)
        # 打印处理之后的消息
        # print(f'第{i}份文档：{source}, \n内容：{content[:100]}')

        # 把内容添加到知识列表中
        knowledge.append({"metadata": doc.metadata, "content": content})
    # 写入Json文件
    with open("./knowledge/knowledge.json", "w", encoding="utf-8") as f:
        json.dump(knowledge, f, ensure_ascii=False, indent=4)
        print("知识保存成功！")