"""利用LangChain按规则做基础文本分割"""
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 递归字符文本分割器

# 加载文档
loader = TextLoader('../data/[all叶]全明星摄像头不要乱扫啊.txt', encoding='utf-8')
docs = loader.load()

# 创建文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    separators=['\n\n', '\n', "。", "，"],  # 定义分隔符
    chunk_size=200,  # 块大小
    chunk_overlap=50  # 相邻块重叠字符
)
# 输出分割后的文档（列表）
chunks = text_splitter.split_text(docs[0].page_content)
for chunk in chunks:
    print(chunk)
    print("--" * 50)