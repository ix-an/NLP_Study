from gensim.models import Word2Vec

# 准备训练语料，每句是一个单词序列
sentences = [
    ['我','爱','自然语言处理'],
    ['自然语言处理','是','人工智能','的','重要','组成部分']
]

# 训练 Word2Vec 模型 -----------------------------------------------
model = Word2Vec(
    sentences,         # 训练语料
    vector_size=10,    # 词向量的维度
    window=2,          # 窗口大小
    min_count=1,       # 最小词频（过滤低频词）
    sg=1               # 模型类型，0 表示 CBOW，1 表示 skip-gram
)
# -----------------------------------------------------------------

# 查看"自然语言处理"的词向量
vector = model.wv['自然语言处理']
print(vector)
print(vector.shape)

# 找出与"人工智能"最相似的词
similar = model.wv.most_similar('人工智能', topn=2)  # topn 表示返回最相似的 n 个词
print(similar)

# 保存模型
model.save('word2vec.model')

# 加载模型
loaded_model = Word2Vec.load('word2vec.model')