# -*- coding: utf-8 -*-
# @Time    : 2024/3/5 11:25
# @Author  : xk
# @File    :泰坦尼克号详细版.py
# @Software: PyCharm
import numpy as np
import pandas as pd
import seaborn as sns
import re
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 一.加载数据
data = pd.read_csv('train_titanic.csv')

# 二.分析数据
# （1）展示数据信息
data.info()
# （2）展示数据前五行
print(data.head())
# （3）展示数据后五行
print(data.tail())
# （4）展示数据结构
print(data.describe())
'''
0   PassengerId  891 
1   Survived     891 
2   Pclass       891 
3   Name         891 
4   Sex          891 
5   Age          714 
6   SibSp        891 
7   Parch        891 
8   Ticket       891 
9   Fare         891 
10  Cabin        204 
11  Embarked     889 
'''

# 三.查看数据是否有异常（有无空值） Age Cabin Embarked有缺失值
# data.isnull().sum()是用来统计数据中每一列的缺失值数量的。
# .isnull()方法会返回一个和原数据形状相同的布尔型数据框，
# 其中缺失值用True表示，非缺失值用False表示。
# sum()方法会将每一列的True值（即缺失值）进行加和，
# 返回的结果是每一列缺失值的数量
# （1）方法1 显示缺失的数量
print(data.isnull().sum())
# （2）方法2 显示缺失字段名（用来打印出数据中含有缺失值的列名）
print(data.columns[pd.isnull(data).sum() > 0])

# 四.如有空值 选择用平均数或众数填充
# （1）找众数
print(data['Embarked'].value_counts())
# （2）找到缺失值的位置
Embarked_null_splace = data['Embarked'].isnull()
# （3）使用loc索引方式进行定位，然后将这些行（2中找到的缺失值的位置）的Embarked列的值，设置为众数（1题中数量最多的）
data.loc[Embarked_null_splace, 'Embarked'] = 'S'
data.info()
# （4）重复上述操作 将Cabin填充
# 用众数和平均值填充不太合适
# 使用假想值进行填充  n--->甲板或底仓
Cabin_null_splace = data['Cabin'].isnull()
data.loc[Cabin_null_splace, 'Cabin'] = 'n'
data.info()

# 五.家庭成员数分析
# （1）分析家庭人数（算自己）
data['family_int'] = data['Parch'] + data['SibSp'] + 1


# （2）定义函数根据家庭成员数划定家庭类型
def fun(n):
    if n <= 1:
        return 'small'
    elif n <= 5:
        return 'middle'
    elif n <= 7:
        return 'big'
    else:
        return 'super'


# （3）同过map映射并添加新字段（家庭成员类型）
data['family_type'] = data['family_int'].map(fun)
data.info()
# （4）绘制家庭成员获救情况统计图
# 通过调用sns.barplot函数并传入相应的参数，
# 可以生成一个以'family_type'为x轴，
# 以'PassengerId'为y轴的柱状图，
# 并根据'Survived'列对柱状图进行分类。
sns.barplot(
    data=data,
    x='family_type',
    y='PassengerId',
    hue='Survived'
)
plt.show()


# 六.通过名字前缀查看获救情况/生死情况（用正则匹配）
# （1）自定义函数用来写正则匹配规则（匹配前缀）
# 可以获取匹配对象中第一个括号内的匹配内容 .group(1)代表匹配第一个括号的内容
def get_name(s):
    return re.search(r'.+,(.+\.+).+', s).group(1)


# （2）添加名字字段（前缀）
data['class_session'] = data['Name'].map(get_name)
data.info()
# （3）绘制不同名字的前缀查看生存的情况
# `sns.countplot`函数是Seaborn库中用于绘制计数柱状图的函数。它可以用来统计每个类别的频数，并将结果可视化成柱状图。
sns.countplot(
    data=data,
    x='class_session',
    hue='Survived'
)
plt.show()

# 七.绘制性别和舱位等级的图例 查看生存的影响（采用多画板绘制）
# （1）用plt.subplots设计画布
fig, axes = plt.subplots(nrows=1, ncols=2)
# `fig`变量存储图形对象，而`axes`变量存储一个包含两个子图的轴对象的数组。
# `ax`参数指定应将绘图绘制到的轴对象(图的位置在画布哪个位置)。
sns.countplot(
    data=data,
    x='Pclass',
    hue='Survived',
    ax=axes[0]
)
sns.countplot(
    data=data,
    x='Sex',
    hue='Survived',
    ax=axes[1]
)
plt.show()

# 八.Age字段填充 使用回归方法预测进行填充
# 思考这些特征能不能对年龄进行预测 内容必须是int类型的
# （1）使用map映射将Fare字段做转整型处理，对船舱名称做字符串切片并取第一个元素处理
data['Fare_int'] = data['Fare'].map(lambda x: int(x))
# 对Cabin字段处理  --->只取仓位的首字母
data['Cabin_type'] = data['Cabin'].map(lambda x: str(x)[0])
# （2）对相关字段做类型处理（独热pd.get_dummies）
# 这里我们本身要 去掉object类型数据  这里我们为什么要还加尼？？
# 因为对数据进行处理
# 对数据进行连续值和离散型处理
'''
Pclass        891 non-null    int64  
Sex           891 non-null    object
family_type   891 non-null    object 
Cabin_type    891 non-null    object 
Embarked      891 non-null    object 
class_session  891 non-null    object
看数据类型   想做独热的话  看obj做独热  一般都是这样
其他的有object  因为有对应的数据
特征维度过多  会有影响
大部分字段不会对生存有影响
例如票价不会 影响你生存
'''
data = pd.get_dummies(
    data=data,
    columns=['Pclass', 'Sex', 'family_type', 'class_session', 'Cabin_type', 'Embarked']
)
# （3）使用算法处理之前要对整型数据做缩放处理，对连续特征进行处理
std_list = ['Fare_int', 'SibSp', 'Parch', 'family_int']
for i in std_list:
    data[i] = StandardScaler().fit_transform(data[[i]])
# 展示缩放后的数据
data.info()
# （4）删除无用特征
'''
PassengerId
Name
Cabin
Ticket
'''
drop_list = ['PassengerId', 'Name', 'Cabin', 'Ticket']
for i in drop_list:
    data.drop(i, axis=1, inplace=True)
data.info()
# （5）划分数据
data_null_age = data['Age'].isnull()
# 1.划分测试集和训练集
data_test = data.loc[data_null_age, :]  # 带有缺失值的数据
data_train = data.loc[~data_null_age, :]  # 不带有缺失值的数据 取反操作
# 2.划分出测试集中的特征和标签 用pop方法
test_y = data_test.pop('Age')
test_x = data_test
# 3.划分出训练集中的特征和标签
train_y = data_train.pop('Age')
train_x = data_train
# （6）创建线性回归模型进行预测填充
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# 1.创建线性回归模型
R_model = Ridge()
# 2.通过网格搜索交叉验证确定alpha参数
model = GridSearchCV(R_model, param_grid={'alpha': [0.1, 0.2, 0.3, 0.4]})
# 3.进行训练
model.fit(train_x, train_y)
# 4.打印最优得分
print('打印最优得分', model.best_score_)
# 5.打印最优参数
print('打印最优参数', model.best_params_)
# 6.利用网格搜索交叉验证确定的参数重新训练模型
R_model = Ridge(alpha=model.best_params_['alpha'])
# 7.训练模型
R_model.fit(train_x, train_y)
# 8.预测数据
test_predict = R_model.predict(test_x)
# 9.打印预测数据
print(test_predict)
# 10.将得到的预测数据填充到Age里面
data.loc[data_null_age, 'Age'] = test_predict
# 11.最后展示填充后的数据
data.info()

# 九.生存预测
# （1）获取特征和标签
y = data.pop('Survived')
x = data
# （2）对特征数据进行特征缩放
x = StandardScaler().fit_transform(x)
# （3）打印特征维度
print('特征维度', x.shape)  # (891, 45) 45维 维度太高 需要降维
# （4）降维
x = PCA(n_components=4).fit_transform(x)
print('降维后的特征维度', x.shape)
# （5）切分训练集测试集
train_x1, test_x1, train_y1, test_y1 = train_test_split(x, y, test_size=0.2)
# （6）使用分类模型 逻辑回归
L_model = LogisticRegression()
# （7）训练模型
L_model.fit(train_x1, train_y1)
# （8）进行预测
test_predict_1 = L_model.predict(test_x1)
print('预测', test_predict_1)
# （9）模型得分
print('模型得分', L_model.score(test_x1, test_y1))

# 十.使用学习曲线进行绘制
# - `estimator`：表示机器学习模型，即要评估的模型。
# - `X`：表示训练数据的特征矩阵。
# - `y`：表示训练数据的目标变量。
# - `train_sizes`：一个数组或迭代器，
# 表示要生成的学习曲线的训练样本数量的比例。
# 在这里，使用`np.linspace(0.1, 1.0, 5)`
# 生成了5个分段均匀的值，表示从10%到100%的训练样本数量。
# np.linspace(0, 1, 5)`会生成一个包含5个元素的等差数列，起始值为0，结束值为1，所以生成的数列为[0.0, 0.25, 0.5, 0.75, 1.0]。

# （1）得到训练得分，测试得分
from sklearn.model_selection import learning_curve

train_size, train_score, test_score = learning_curve(
    estimator=L_model,
    X=x,
    y=y,
    train_sizes=np.linspace(0.1, 1.0, 5)
)
print('训练得分', train_score)
print('测试得分', test_score)

# （2）求均值 求标准差
train_score_mean = np.mean(train_score, axis=1)
train_score_std = np.std(train_score, axis=1)
test_score_mean = np.mean(test_score, axis=1)
test_score_std = np.std(test_score, axis=1)

# （3）绘制图
# 在这段代码中，plt.plot()函数用于绘制折线图。具体参数的含义如下：
# - train_size：x轴的数据，表示训练样本的大小。
# - train_score_mean：y轴的数据，表示训练得分的平均值。
# - 'ro-'：表示绘制红色的圆形点和实线连接的折线。
# - label='训练上的得分'：为折线添加标签，用于图例展示。
# 1.绘制训练上的得分
plt.plot(
    train_size,
    train_score_mean,
    'ro-',
    label='训练上的得分'
)
# 2.绘制阴影部分 查看数据是否是过拟合状态
plt.fill_between(
    x=train_size,
    y1=train_score_mean + train_score_std,  # 上阴影边界
    y2=train_score_mean - train_score_std,  # 下阴影边界
    # 透明度
    alpha=0.9
)
# 3.绘制测试上的得分
plt.plot(
    train_size,
    test_score_mean,
    'bo-',
    label='测试上的得分'
)
# 4.绘制阴影部分 查看数据是否过拟合状态
plt.fill_between(
    x=train_size,
    y1=test_score_mean + test_score_std,
    y2=test_score_mean - test_score_std,
    alpha=0.9
)
plt.title('学习曲线')
plt.xlabel('训练数据')
plt.ylabel('模型得分')
plt.legend()
plt.show()
