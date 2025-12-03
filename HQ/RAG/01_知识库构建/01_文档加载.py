
# 加载txt文档
from langchain_community.document_loaders import TextLoader

loader = TextLoader('../data/[all叶]全明星摄像头不要乱扫啊.txt', encoding='utf-8')
# 返回一个Document对象
documents = loader.load()
print(documents[0].metadata)  # 文档元数据
print(documents[0].page_content)  # 文档内容

"""---------------------------------------------------------------------------"""

# 加载CSV文档
from langchain_community.document_loaders import CSVLoader
loader = CSVLoader('../data/黑悟空.csv', encoding='utf-8')
documents = loader.load()

"""---------------------------------------------------------------------------"""

# 加载Word文档
from langchain_community.document_loaders import Docx2txtLoader

"""---------------------------------------------------------------------------"""

# 加载PDF 文档
from langchain_community.document_loaders import PyPDFLoader
# PDF是图片格式，不用填写encoding
loader = PyPDFLoader('../data/[all叶]全明星摄像头不要乱扫啊.pdf')

"""---------------------------------------------------------------------------"""

# 加载JSON文档
from langchain_community.document_loaders import JSONLoader
# 标准的JSON文件必须单行，否则会报错
# Document对象的page_content属性需要是字符串类型
loader = JSONLoader(
    file_path='../data/person.json',
    jq_schema='.[] | tostring'  # jq 的管道语法，将提取出的元素转换为字符串
)
documents = loader.load()
print(documents)

loader = JSONLoader(
    file_path='../data/person.json',
    # 提取name和sex字段，创建自定义格式的字符串（字符串插值）
    jq_schema=r'.[]|"姓名:\(.name), 性别:\(.sex)"'
)
documents = loader.load()
print(documents[0].page_content)  # 姓名:karen, 性别:female