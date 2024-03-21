# -*- coding: utf-8 -*-
# Time : 2024/2/26 12:04
# Author : chen
# Software: PyCharm
# File : 共享单车项目代码实现.py
import warnings

warnings.filterwarnings('ignore')
import time
import datetime
import calendar  # 年历
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# plt.rcParams['font.sans_seif'] = ['SimHei']
# 读取数据
data = pd.read_csv('train.csv')
# 打印数据的基本信息  数据检视三板斧
data.info()
# 打印数据的前五行
print(data.head())
# 打印数据的后五行
print(data.tail())
# 打印数据的基本结构
print(data.describe())
'''========时间处理======='''


# 数据类型字段处理 时间字段  datetime  ‘2011-01-01’
# 把数据类型字段转换成时间组

# t1 = datetime.datetime.strptime('2011-01-01 00:00:00','%Y-%m-%d %H:%M:%S')
# t2 = datetime.datetime.fromisoformat('2011-01-01 00:00:00')
# print(datetime.datetime.fromisoformat('2011-01-01 00:00:00'))
# 年 year  月  month   天  day  小时  hour   分钟  minist   秒 second
# print(t2)
# 获取月份信息
def get_month(mt):
    # 时间组的形式类型
    month = datetime.datetime.fromisoformat(mt)
    return month


# 获取天的信息
def get_day(da):
    day = datetime.datetime.fromisoformat(da).day
    return day


# 获取小时信息
def get_hour(gu):
    hour = datetime.datetime.fromisoformat(gu).hour
    return hour


# 获取相对应的月份信息
def get_month_name(gmn):
    month = datetime.datetime.fromisoformat(gmn)
    month_name = month.month
    return calendar.month_name[month_name]


# 获取对应的星期几的信息
def get_weekday_name(wn):
    weekday = datetime.datetime.fromisoformat(wn)
    weekday_name = weekday.weekday()
    return calendar.day_name[weekday_name]


# 将得到的月份  天   小时  返回/添加数据当中  --->data
data['month'] = data['datetime'].map(get_month)
data['day'] = data['datetime'].map(get_day)
data['hour'] = data['datetime'].map(get_hour)
# 检查数据是否添加成功
data.info()

# 将月份名字和星期名字也相对应添加到数据data中
data['month_name'] = data['datetime'].map(get_month_name)
data['day_name'] = data['datetime'].map(get_weekday_name)
# 检查数据是否添加成功
data.info()

# 绘图查看骑行数量 点线图
# 每个小时对应的骑行量数据
sns.pointplot(
    data=data,
    x='hour',
    y='count'
)
plt.show()


# 每个月对应的骑行量数据
# sns.pointplot(
#     data=data,
#     x='month',
#     y='count'
# )
# plt.show()
# 对数据进行分层  利用数据分析
def fo(hour):
    if hour >= 0 and hour <= 6:
        return 1
    elif hour >= 7 and hour <= 10:
        return 2
    elif hour >= 11 and hour <= 15:
        return 3
    elif hour >= 16 and hour <= 20:
        return 4
    else:
        return 5


data['hour_type'] = data.hour.map(fo)
data.info()
sns.pointplot(data=data, x='hour_type', y='count')
plt.show()

# 多画板绘制  子图  噪声处理
# fig ==>画板
# axes ==>坐标
fig, axes = plt.subplots(nrows=1, ncols=2)  # nrows-->多少行 ncols-->多少列
sns.boxplot(
    data=data,
    y='count',
    ax=axes[0]
)
sns.boxplot(
    data=data,
    x='season',
    y='count',
    ax=axes[1]
)
plt.show()

# 箱型图中有很多的黑点 代表的是异常数据  所以要处理异常数据
# 1获取骑行量的数据  查看骑行量的异常值
# 索引
# iloc==  用数据的索引或者行索引进行异常取值 iloc[3,11]  3行11列
# loc==  字段名或者行名去取值
get_counts = data.loc[:, ['count']]
print(get_counts)
# 将拿到的骑行的数据进行异常值处理 --》找异常值/找噪声
# 查找噪声值的公式：(数据点  - 均值超过该标准差的三倍则视为噪声)
# 计算骑行量的均值
get_counts_mean = np.mean(get_counts)
# 计算骑行量的标准差
get_counts_std = np.std(get_counts)
# 判断噪声的条件  数据点 - 均值 > 标准差的三倍
# 计算每个骑行量与平均值之间的距离点
dist_counts = get_counts - get_counts_mean
# 将距离值和大于3倍的标准差比较  大于的就是异常值  其他都是正常值
noise = dist_counts > 3 * get_counts_std
# 如何拿出异常值的数据  异常数据的格式都应该是什么？ true

noise_val_opt = noise.values.flatten()
print(noise_val_opt)
# 数据的取反操作
# ~--->代表取反
# print(~noise_val_opt)
get_data_good = data.loc[~noise_val_opt, :]
get_data_good.info()

# 数据分析
# 分析天气跟骑行量之间的关系
# 要制作热力图
get_data_good_list = [
    'count', 'weather', 'temp',
    'atemp', 'windspeed', 'casual', 'registered'
]
dd_good_list = data.loc[:, get_data_good_list]
# 计算皮尔逊系数
dd_good_list_corr = dd_good_list.corr()
sns.heatmap(
    data=dd_good_list_corr,
    annot=True
)
plt.show()
