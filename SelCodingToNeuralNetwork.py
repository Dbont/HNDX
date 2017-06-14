# load packages~~~~~~~~~~yeah
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Input
import numpy as np
import pandas as pd
import pickle, pprint # 万一程序崩溃了咋办
import warnings # just for pretty
warnings.filterwarnings('ignore')
import os # for path

# 加载数据，按时间分成几组，为做类似交叉验证准备数据
# 传入参数分别为：分组数、测试集比例、工作路径、目录
# 输出数据分比为：训练集X、训练集Y、测试集X、测试集Y、列数n(包含index和Y)、数据的分组数、文件名
def LoadPreData(GroupNumber = 5, Ratio = 0.75, path = r'E:\Work\HNDX\DATA\Work', 
                data = '4day_000300_minute.csv'):
    # 设置路径，读取数据
    os.chdir(path)
    df0 = pd.read_csv(data)
    # 读取数据的行数与列数
    m = df0.shape[0] # 行数
    n = df0.shape[1] # 列数
    # 按照组数、比例，分出间断点
    x = round(m / GroupNumber * Ratio)
    # 把分出的数据分装在这四个列表中
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    # 下面的循环会完成上一步的目标
    for i in range(1, GroupNumber+1):
        n0 = round(((i - 1) * m)/ GroupNumber)
        n1 = round(i * m / GroupNumber)
        df = df0[n0: n1]
        # 把pandas数据转化为numpy数据，因为keras只能识别numpy数据
        numpyMatrix = df.as_matrix()
        # 依次添加数据
        X_train.append(numpyMatrix[0 : x, 1 : n - 1])
        Y_train.append(numpyMatrix[0 : x, n - 1])
        X_test.append(numpyMatrix[x :, 1 : n - 1])
        Y_test.append(numpyMatrix[x :, n - 1])
    return X_train, Y_train, X_test, Y_test, n, GroupNumber, data

# 这一步要先进行自编码学习，把特征降到10维
# 传入参数分别为第一层神经元数、需要降低到的维度、数据分组数、训练数据X、测试数据X
# 传出数据为自编码模型运行完后的训练集、测试集的预测数据，
def SelCoding(n_in = n - 2, encoding_dim = 10, GroupNumber = GroupNumber, 
              X_train = X_train, X_test = X_test, data = data):
    # 传入数据
    input_img = Input(shape=(n_in,))
    # 编码层  
    encoded2 = Dense(round(n_in * 1 / 2), activation='relu')(input_img)  
    #encoded3 = Dense(round(n_in * 1 / 3), activation='relu')(encoded2)  
    encoder_output = Dense(encoding_dim)(encoded2)
    # 解码层  
    decoded4 = Dense(round(n_in * 1 / 2), activation='relu')(encoder_output)  
    #decoded6 = Dense(round(n_in * 2 / 3), activation='relu')(decoded5)   
    decoded5 = Dense(n_in, activation='tanh')(decoded4)
    # 构建自编码模型  
    autoencoder = Model(inputs=input_img, outputs=decoded5)
    # 构建编码模型  
    encoder = Model(inputs=input_img, outputs=encoder_output)
    # compile autoencoder  
    autoencoder.compile(optimizer='adam', loss='mse')
    # 按照分组数据分别训练
    X_train_encoded = []
    X_test_encoded = []
    for i in range(GroupNumber):
        autoencoder.fit(X_train[i], X_train[i], epochs = 2, batch_size=20, shuffle=True)
        # 把训练完，降维后的数据保存起来
        encoder_OUT = Model(input = input_img, output=encoder_output)
        # 保存训练集降维后的预测数据
        X_train_encoded0 = encoder_OUT.predict(X_train[i])
        X_train_encoded.append(X_train_encoded0)
        # 保存测试集降维后的预测数据
        X_test_encoded0 = encoder_OUT.predict(X_test[i])
        X_test_encoded.append(X_test_encoded0)
    # 把数据保存在.pkl文件中
    name = data + '_SelCoding_X10dim.pkl'
    output = open(name, 'wb')
    pickle.dump(X_train_encoded, output)
    pickle.dump(X_test_encoded, output)
    output.close()
    # 以下为调用语法，需要调用两次
    #pkl_file = open(name, 'rb')
    #data = pickle.load(pkl_file)
    #pprint.pprint(data)
    #data = pickle.load(pkl_file)
    #pprint.pprint(data)
    return X_train_encoded, X_test_encoded, encoding_dim

def NeuralNetwork(n_in = encoding_dim, X_train = X_train_encoded, Y_train = Y_train, 
                  X_test = X_test_encoded, Y_test = Y_test, 
                  GroupNumber = GroupNumber, data = data):
    # 建立神经网络模型
    model = Sequential()
    model.add(Dense(20, input_dim=n_in),)
    model.add(Activation('relu'))
    model.add(Dense(20))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    # 编译模型（初始化）
    model.compile(optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy'])
    # 训练模型
    score_all = []
    for i in range(GroupNumber):
        model.fit(X_train[i], Y_train[i], nb_epoch = 20, batch_size = 20)
        score = model.evaluate(X_test[i], Y_test[i], batch_size = 20)
        score_all.append(score[1])

    name = data + '_NeuralNetwork10-20-20-1_AccuracyRate.pkl'
    output = open(name, 'wb')
    pickle.dump(score_all, output)
    output.close()
    # Step4: show the outcome
    pkl_file = open(name, 'rb')
    data = pickle.load(pkl_file)
    pprint.pprint(data)


################################################################################
# 以下为具体执行代码

[X_train, Y_train, X_test, Y_test, n, GroupNumber, data] = LoadPreData()

[X_train_encoded, X_test_encoded, encoding_dim] = SelCoding()

NeuralNetwork()