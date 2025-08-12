"""数据预处理：读取源文件，把文字进行编码处理"""
import torch


"""公共参数"""
# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 源文件路径 -> 每行包含标签（正/负）和对应的评论文本
org_file_path = './datasets/hotel_discuss2.csv'
# 字典文件路径 -> 给每个出现的文字分配一个唯一的数字编码（索引）
dict_file_path = './datasets/word_index.txt'
# 编码文件路径 -> 字典中数字编码替换源文件文字后生成的文件
encode_file_path = './datasets/encode.txt'
# 过滤字符
filtering_characters = ["。"]
# 字典
mydict = {}    # 存储 文字->数字 的映射


"""读取源文件，把文字处理为字典并存储"""
def read_file():
    count = 1    # 0保留作 padding
    print("正在处理数据...")
    with open(org_file_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
        for line in lines:
            strip_line = line.strip()    # 去掉首尾空格
            for ch in strip_line:
                if ch in filtering_characters:
                    continue    # 跳过过滤字符
                if ch in mydict:
                    continue    # 跳过已经在字典中的字符
                else:
                    mydict[ch] = count    # 不在字典，则添加
                    count += 1    # 字符数量

        mydict["<UNK>"] = count    # 未知字符编码

        print("数据处理完成")
        print("准备开始保存数据...")
        with open(dict_file_path, 'w', encoding='utf-8-sig') as f:
            f.write(str(mydict))
            print("数据保存成功")


"""对样本中评论部分进行编码"""
def encode_sample():
    # 编码处理
    print("准备编码处理...")
    with open(org_file_path, 'r', encoding='utf-8-sig') as f:
        with open(encode_file_path, 'w', encoding='utf-8-sig') as fw:
            for line in f.readlines():    # 遍历原始样本每一行
                label = line[0]    # 标签
                comment = line[1:-1]    # 评论
                for ch in comment:    # 遍历评论，对每个文字进行编码
                    if ch in filtering_characters:
                        continue    # 跳过过滤字符
                    else:
                        fw.write(str(mydict[ch]))    # 写入编码值
                        fw.write(",")    # 每个文字编码后用逗号隔开
                fw.write("\t" + str(label) + "\n")    # 评论编码后，用制表符隔开标签
    print("编码处理完成")


"""主函数"""
def main():
    read_file()  # 得到 word_index.txt
    encode_sample()  # 得到 encode.txt

"""运行主函数"""
if __name__ == '__main__':
    main()














