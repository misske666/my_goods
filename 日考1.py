import tensorflow as tf

# 1.使用tensorflow2.0完成以下操作（每小题10分）
# (1)矩阵创建
# ①创建一个维度为3*3的全1矩阵
ones = tf.ones(shape=[3, 3])

# ②使用range，创建一个1-9的1阶张量
tf_range = tf.range(start=1, limit=10)  # 其中从`start`开始（包括`start`）逐步增加，直到`limit`（不包括`limit`）。这个函数类似于Python中的`range()`函数。

# ③打印上题的维度
tf.print(tf_range.shape)

# ④将上题维度修改为3,1,3
tr_reshape = tf.reshape(tf_range, shape=[3, 1, 3])

# ⑤使用函数，去除维度中函数1的维度
tf_squeeze = tf.squeeze(tr_reshape)
tf.print(tf_squeeze.shape)

# (2)切片及其他
# ①使用1-9的向量，使用切片，打印3,4,5,6  1，2，3，4，5，6，7，8，9 a[2:5]
range_ = tf_range[2: 6]
tf.print(range_)

# ②打印上题向量的均值
tf.print(tf.reduce_mean(range_))

# ③创见一个2行2列的标准正态分布矩阵
normal_ = tf.random.normal(shape=[2, 2])
tf.print(normal_)

# ④创建一个2行2列的全0矩阵
zeros_ = tf.zeros(shape=[2, 2])
tf.print(zeros_)

# ⑤将3,4问的结果拼接成一个4行2列的结果
tf.print(tf.concat([normal_, zeros_], axis=0))