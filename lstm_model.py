# 要求：
# 建立falsk框架，将下方代码封装到falsk框架中，路由自定，通过postman进行测试
from flask import Flask
# (1)从tensorflow.keras中导入所需的模块
from tensorflow.keras import Sequential, layers, optimizers, losses, activations, utils

# (2)定义字符串为"applebanana"，将字符串作为路由参数进行传参，要求使用post方式
sample = 'applebanana'
# (3)要有参数非空判断，如果未传参数给出友好提示

# (4)获取前n-1个字符作为x_data，获取后n-1个字符作为y_data
x_data = sample[:-1]
y_data = sample[1:]
# (5)使用set函数，去除重复的字符
sample_kinds = set(sample)
# (6)获取去重后的字符个数，作为独热编码的维度
sample_dim = len(sample_kinds)
# (7)获取输入序列的长度
seq_len = len(x_data)
# (8)创建字符索引到字符的映射
char_to_int = {j: i for i, j in enumerate(sample_kinds)}
# (9)创建字符到字符索引的映射
int_to_char = {i: j for i, j in enumerate(sample_kinds)}
# (10)将x_data中的字符映射为对应的索引
x_data = [char_to_int[i] for i in x_data]
# (11)将y_data中的字符映射为对应的索引
y_data = [char_to_int[i] for i in y_data]
# (12)对x_data进行独热编码，并调整形状为(-1, s_len, onehot_dim)
x_data = utils.to_categorical(x_data, sample_dim).reshape(-1, seq_len, sample_dim)
# (13)对y_data进行独热编码，并调整形状为(-1, s_len, onehot_dim)
y_data = utils.to_categorical(y_data, sample_dim).reshape(-1, seq_len, sample_dim)


def transform_data(input_str):
    input_str = [char_to_int[i] for i in input_str]
    input_str = utils.to_categorical(input_str, sample_dim).reshape(-1, seq_len, sample_dim)
    return input_str


# (14)创建Sequential模型
model = Sequential([
    # (15)添加具有64个神经元并返回完整序列的LSTM层
    # (16)添加具有64个神经元并返回完整序列的LSTM层
    # (17)添加具有onehot_dim个单元并使用softmax激活函数的全连接层
    layers.LSTM(units=64, return_sequences=True),
    layers.LSTM(units=64, return_sequences=True),
    layers.Dense(units=sample_dim, activation=activations.softmax)
])
# (18)构建模型，指定输入数据的形状
model.build(input_shape=(None, seq_len, sample_dim))
# (19)打印模型结构信息
model.summary()
# (20)编译模型，指定优化器Adam、损失函数categorical_crossentropy和评估指标acc
model.compile(optimizer=optimizers.Adam(), loss=losses.categorical_crossentropy, metrics='acc')
# (21)训练模型
model.fit(x_data, y_data, epochs=1000, batch_size=100)
# (22)获取模型对x_data的预测结果
predict = model.predict_classes(x_data)
# (23)将预测结果转换为字符并打印出来
target = ''.join([int_to_char[i] for i in predict[0]])


def int_res_to_char(a):
    return ''.join([int_to_char[i] for i in a])


model.save('lstm.h5')
# (24)将以上代码上传至Git仓库
# (24)将以上代码上传至Git仓库
