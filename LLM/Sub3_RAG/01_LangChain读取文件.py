
from langchain_community.document_loaders import TextLoader  # 支持各种格式的数据

if __name__ == '__main__':
    loader = TextLoader("./data/仙某某.txt", encoding="utf-8")
    documents = loader.load()
    print(documents)

