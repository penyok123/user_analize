import pickle
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

# train = pd.read_csv("../data/train.csv")

datafile = open('./data_sikuquanshus.pkl', 'rb')
print("load  data  by pkl")
# text of train
X_train = pickle.load(datafile)
X_test=pickle.load(datafile)
# text of test
Y_train = pickle.load(datafile)
Y_test=pickle.load(datafile)
word_to_index = pickle.load(datafile)
index_to_word = pickle.load(datafile)
word_to_vec_map = pickle.load(datafile)
datafile.close()

#################


# 我们使用的预训练的 word embedding 是 40 万个单词的训练结果，它们的特征维数是 50
def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    创建一个 Keras 的 Embedding() 层，并且加载之前已经训练好的 embedding
    """
    print("训练embeddings")
    # 词典中单词的个数+1，+1是 keras 模型的训练要求，没有什么其他含义
    vocab_len = len(word_to_index) + 1
    # 获取单词的特征维数，随便找个单词就行了
    emb_dim = word_to_vec_map["错"].shape[0]

    # 将 embedding 矩阵初始化为全 0 的，大小为 (vocab_len, emb_dim)
    emb_matrix = np.zeros((vocab_len, emb_dim))  #(19529 ,300)
    # 将 emb_matrix 的行号当做单词的编号，然后将这个单词的 embedding 放到这一行，这样就把预训练的 embedding 加载进来了
    # 注意，由于单词编号是从 1 开始的，所以行 0 是没有 embedding 的，这就是为什么前面要 +1
    for word, index in word_to_index.items():  #(19529 ,300)
        emb_matrix[index, :] = word_to_vec_map[word]

    # 创建 Keras 的Embedding 层
    embedding_layer = Embedding(input_dim=vocab_len, output_dim=emb_dim, trainable=True)

    # build embedding layer，在设置 embedding layer 的权重的时候，这一步是必须的
    embedding_layer.build((None,))

    # 将 emb_matrix 设置为 embedding_layer 的权重。
    # 到这里为止我们就创建了一个预训练好的 embedding layer
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer

# 其他所有的分类模型可以基于这个函数进行创建
def mother_model(input_shape, word_to_vec_map, word_to_index):
    """
    返回：一个 Keras 的模型
    参数:
    input_shape -- MAX_COMMENT_TEXT_SEQ
    word_to_vec_map
    word_to_index
    """
    # 创建输入层，输入的是句子的单词编号列表
    sentence_indices = Input(shape=input_shape, dtype=np.int32)   #(?,1000)
    # 创建 word embedding 层
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)

    # 句子编号列表进入 embedding_layer 之后会返回对应的 embeddings
    embeddings = embedding_layer(sentence_indices)  #(?,1000,300)
    dr_r = 0.5

    X = BatchNormalization()(embeddings)
    X = LSTM(128, return_sequences=True)(X)
    X = Dropout(dr_r)(X)
    X = BatchNormalization()(X)
    X, _, __ = LSTM(128, return_state=True)(X)
    X = Dropout(dr_r)(X)

    X = BatchNormalization()(X)
    X = Dense(64, activation='relu')(X)

    X = Dropout(dr_r)(X)

    X = BatchNormalization()(X)
    X = Dense(6, activation='sigmoid')(X)

    model = Model(inputs=sentence_indices, outputs=X)

    return model


MAX_COMMENT_TEXT_SEQ = 1000
toxic_model = mother_model((MAX_COMMENT_TEXT_SEQ,), word_to_vec_map, word_to_index)
toxic_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model_dir = './models'
filepath = model_dir + '/model-{epoch:02d}.h5'
checkpoint = ModelCheckpoint(filepath,monitor='val_loss',save_best_only=True, verbose=1)
callbacks_list = [checkpoint]
train_result = toxic_model.fit(X_train, Y_train,
                    epochs=1,
                    batch_size=1000,
                    validation_split=0.07,
                    callbacks = callbacks_list)
loss,accuracy=toxic_model.evaluate(X_test,Y_test)
print("loss",loss)
print("accuracy",accuracy)
