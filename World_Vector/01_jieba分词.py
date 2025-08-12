import jieba
from jieba.analyse import extract_tags


# 基础分词：使用lcut系列，保存为列表
def cut_text():
    text = "南京市长江大桥"

    seg_list = jieba.cut(text)                  # 精确模式
    print("精确模式：" + "/".join(seg_list))

    seg_list = jieba.cut(text, cut_all=True)    # 全模式
    print("全模式：" + "/".join(seg_list))

    seg_list = jieba.cut_for_search(text)       # 搜索引擎模式
    print("搜索引擎模式：" + "/".join(seg_list))


# 添加新词
def add_word_test():
    """是向分词词典里添加新词，会改变分词结果"""
    text = "我来到国王大道A点高台"
    org_list = jieba.lcut(text)
    print("原始分词：" + "/".join(org_list))
    # 向词典添加新词
    jieba.add_word("国王大道A点高台")
    new_list = jieba.lcut(text)
    print("新分词：" + "/".join(new_list))


# 加载自定义词典
def  load_dict_test():
    jieba.load_userdict("my_dict.txt")
    text = "莱因哈特美丽"
    result = jieba.lcut(text)
    print(result)


# 关键词提取
from jieba.analyse import extract_tags
def  keywords_test():
    text = "白鸽在梦里安睡 红的绣花 眼泪载满花舱"

    # 不提取权重 -> 返回关键词列表
    res = extract_tags(text, topK=3)

    # 提取权重 -> 返回关键词和权重值组成的元组列表
    res_weight = extract_tags(text, topK=3, withWeight=True)

    print(res)
    print(res_weight)


# 添加停用词
def stop_word_tset():
    text = "我演过依依不舍的人，压抑着嫉妒笑着祝福的人"
    # 添加词典
    adds = {"笑着祝福", "依依不舍"}
    for i in adds:
        jieba.add_word(i)

    org_list = jieba.lcut(text)
    print("未使用停用词：", org_list)

    # 设置停用词集合
    stopwords = {"的", "是", "和", "，", "！", "人"}
    # 过滤停用词
    new_list = [i for i in org_list if i not in stopwords]
    print("使用停用词：", new_list)



# if __name__ == '__main__':
#     cut_text()
#     add_word_test()
#     load_dict_test()
#     keywords_test()
#     stop_word_tset()
