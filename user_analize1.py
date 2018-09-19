import numpy as np
import pickle

import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
# 从整个训练集数据集中抽取部分数据作为训练模型的训练集数据和测试集数据，并且指定要使用的目标变量
def input_data(train_file, divide_number, end_number, tags):
    train_words = []
    train_tags = []
    test_words = []
    test_tags = []

    with open(train_file, "r", encoding="gb18030") as f:
        text = f.readlines()

        # 构建训练集数据
        train_data = text[:divide_number]
        for single_query in train_data:
            # 先将所有的字段分割
            single_query_list = single_query.split(" ")
            # 去除ID字段
            single_query_list.pop(0)
            # 标签确定的情况下构建样本
            if single_query_list[tags] != '0':
                # 构建训练集样本的目标变量
                train_tags.append(single_query_list[tags])
                # 删除3个目标变量 剩下关键词
                single_query_list.pop(0)
                single_query_list.pop(0)
                single_query_list.pop(0)
                train_words.append(
                    (str(single_query_list)).replace(',', ' ').replace('\'', '').lstrip('[').rstrip(']').replace('\\n',
                                                                                                                 ''))

        # 构建测试集数据 构建的方法和训练集数据的构建是一样的
        test_data = text[divide_number:end_number]
        for single_query in test_data:
            single_query_list = single_query.split(" ")
            single_query_list.pop(0)
            if single_query_list[tags] != 0:
                test_tags.append(single_query_list[tags])
                single_query_list.pop(0)
                single_query_list.pop(0)
                single_query_list.pop(0)
                test_words.append(
                    (str(single_query_list)).replace(",", ' ').replace('\'', '').lstrip('[').rstrip(']').replace('\\n',
                                                                                                                 ''))
    print('input_data.done !')
    return train_words, train_tags, test_words, test_tags

def encodes_tags(train_tags,test_tags):
    label = LabelEncoder()
    train_tags = label.fit_transform(train_tags)
    one = OneHotEncoder()
    train_tags = one.fit_transform(train_tags.reshape(-1, 1)).toarray()

    label = LabelEncoder()
    test_tags = label.fit_transform(test_tags)
    one = OneHotEncoder()
    test_tags = one.fit_transform(test_tags.reshape(-1, 1)).toarray()

    print("train_tags",train_tags)
    return train_tags,test_tags


def read_embedding(embedding_file):
    with open(embedding_file,"r",encoding="utf-8") as f:
        #将单词保存到一个集合中
        words=set()
        #将单词与embeddings的映射保存到一个字典中
        words_to_index = {}
        index_to_words = {}
        word_to_vec_map={}
        for line in f:
            line=line.strip().split()
            #列表的第一个元素是单词
            curr_word=line[0]
            #将单词加入到集合中
            words.add(curr_word)
            #列表的其余元素是embeddings 将单词与embeddings进行映射，然后保存到字典中
            word_to_vec_map[curr_word]=np.array(line[1:],dtype=np.float64)
            #将单词进行编号 编号从1 开始
            i=1
            for w in sorted(words):
                #创建映射 可以是单词 value是编号
                words_to_index[w]=i
                #创建映射 key是编号 value 是单词
                index_to_words[i]=w
                #编号递增
                i=i+1
        #将生成的三个映射返回
    return words_to_index,index_to_words,word_to_vec_map

# 1.5 将单词转换为编号
def sentences_to_indices(X,words_to_index):
    X_indices=[]
    for word in X:
        try:
            X_indices.append(words_to_index[word])
        except:
            pass
    return X_indices

def get_max(train_word_index):
    list=[]
    for i in range(len(train_word_index)):
        list.append(len(train_word_index[i]))
    plt.hist(list,bins=2000,normed=1)
    plt.show()
# pad_sequences 函数用于将数字列表进行填充
# 如果列表长度大于最大长度，那么将列表进行裁剪，如果列表长度小于最大长度，那么将列表补充到最大长度，并且默认填充0
#
from keras.preprocessing.sequence import pad_sequences
MAX_COMMENT_TEXT_SEQ = 1000
def get_keras_data(dataset):
    s = pad_sequences(dataset, maxlen=MAX_COMMENT_TEXT_SEQ, padding='post')
    return s

def sgns_embedding(train_words,test_words,embedding_file):
    train_word_index=[]
    test_word_index=[]

    words_to_index,index_to_words,word_to_vec_map=read_embedding(embedding_file)
    print("已经得到了words_to_index,index_to_words,word_to_vec_map")
    for i in range(len(train_words)):
        train_word_index.append(sentences_to_indices(train_words[i],words_to_index))
    for i in range(len(test_words)):
        test_word_index.append(sentences_to_indices(test_words[i],words_to_index))
    print(train_word_index)
    print("已经把汉字变成了编号")
    #获取最大的长度
    train_max=get_max(train_word_index)
    # 将训练集数据的文本编号列表进行填充，并且提取出来
    train_words = get_keras_data(train_word_index)
    # 将测试集数据的文本编号列表进行填充，并且提取出来
    test_words = get_keras_data(test_word_index)
    print(train_words)
    print("进行填充了")
    return train_words,test_words,words_to_index,index_to_words,word_to_vec_map


def save_model(train_words, train_tags, test_words, test_tags,words_to_index,index_to_words,word_to_vec_map):
    datafile = open("data_sikuquanshus.pkl", "wb")
    pickle.dump(train_words, datafile)
    pickle.dump(test_words, datafile)
    pickle.dump(train_tags, datafile)
    pickle.dump(test_tags, datafile)
    pickle.dump(words_to_index, datafile)
    pickle.dump(index_to_words, datafile)
    pickle.dump(word_to_vec_map, datafile)
    datafile.close()


# 对指定的目标变量 "F:\中文词向量\sgns.literature.word"进行测试
def test_single(tags):
    train_file = "train_data_fenci.txt"
    embedding_file = "F:\中文词向量\sgns.literature.word"
    # 测试集起始位置
    divide_number = 15500
    # 测试集终止位置
    end_number = 17633

    print("file:" + train_file)
    print("tags:%d" % tags)

    # 将数据集分为训练与测试 获取训练与测试数据的标签
    train_words, train_tags, test_words, test_tags = input_data(train_file, divide_number, end_number, tags)
    train_tags,test_tags=encodes_tags(train_tags,test_tags)
    #获取词嵌入中文词库 并把关键词转成编号
    train_words, test_words,words_to_index,index_to_words,word_to_vec_map=sgns_embedding(train_words, test_words, embedding_file)
    #保存处理好的数据
    save_model(train_words, train_tags, test_words, test_tags,words_to_index,index_to_words,word_to_vec_map)

def test():
    # 标签（年龄性别学历） 卡方选取后的维数 主题个数
    # 0 对应age
    # 1 对应 Gender
    # 2 对应Education
    #我们分别对这山歌标签作为我们的目标变量进行训练
    test_single(0)
    #test_single(1)
    # test_single(2)

import sys
def main():
    #如果第一个参数是test 那么对3个目标变量分别进行测试 看看分类效果如何
    test()

if __name__ == '__main__':
    main()