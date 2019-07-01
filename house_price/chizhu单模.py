# coding: utf-8
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import KFold

# 载入数据
train = pd.read_csv("data/train_data.csv")
test = pd.read_csv("data/test_a.csv")

# 数据清洗
# train=train[train["tradeMoney"]<25000]
# train=train[train["area"]<200]
# train=train[train["area"]>12]
# train=train[(train["tradeMoney"]<18000)&(train["tradeMoney"]>100) ]
# train=train[train["area"]<200]
# train=train[train["area"]>10]
# train = train [train['totalFloor'] <80]
# train = train [train['tradeMeanPrice'] <100000]
# train = train [train['remainNewNum'] <4000]

# 合并train,test
test['tradeMoney'] = -1
data = pd.concat([train,test])

# 查看“类别型”特征
for col in data.columns:
    if data[col].dtype=="object":
        print(col)


data['local'] = data['region']+"_"+data['plate']+"_"+data['communityName']
# 'houseToward'值替换，'暂无数据'变为'南'
data['houseToward']=data['houseToward'].apply(lambda x:x.replace("暂无数据","南"),1)


import time


# 定义timestamp生成函数
def datetime_timestamp(dt):
    s = time.mktime(time.strptime(dt, '%Y/%m/%d'))
    return s


# 定义get_feat函数，创建或修改各种特征
def get_feat(data):
    data['pv'] = data['pv'].fillna(data['pv'].mean())
    data['uv'] = data['uv'].fillna(data['uv'].mean())
    data['rentType'] = data['rentType'].map({"整租": 1, "合租": 2})
    data['room_num'] = data['houseType'].apply(lambda x: int(x[0]), 1)
    data['hall_num'] = data['houseType'].apply(lambda x: int(x[2]), 1)
    data['toilet_num'] = data['houseType'].apply(lambda x: int(x[4]), 1)
    data['total_num'] = data['room_num'] + data['toilet_num'] + data['hall_num']
    data['houseFloor'] = data['houseFloor'].map({"低": 0, "中": 1, "高": 2})
    data['houseDecoration'] = data['houseDecoration'].map({"其他": 0, "毛坯": 1, "简装": 2, "精装": 3})
    data['communityName_freq'] = data['communityName'].map(
        data['communityName'].value_counts().rank() / len(data['communityName'].unique()))
    data['region'] = data['region'].map(data['region'].value_counts().rank() / len(data['region'].unique()))
    data['plate'] = data['plate'].map(data['plate'].value_counts().rank() / len(data['plate'].unique()))
    data['local'] = data['local'].map(data['local'].value_counts().rank() / len(data['local'].unique()))
    #     data['houseType']=data['houseType'].map(data['houseType'].value_counts().rank()/len(data['houseType'].unique()))
    data['buildYear'] = data['buildYear'].apply(lambda x: x.replace("暂无信息", "1970"), 1)
    data['buildYear'] = data['buildYear'].astype(int)
    data['year'] = data['tradeTime'].apply(lambda x: int(x.split("/")[0]), 1)
    data['month'] = data['tradeTime'].apply(lambda x: int(x.split("/")[1]), 1)
    data['day'] = data['tradeTime'].apply(lambda x: int(x.split("/")[2]), 1)
    #     data['build_time']=data['year']-data['buildYear']
    data['tradeTime'] = data['tradeTime'].apply(lambda x: datetime_timestamp(x), 1)
    for col in ['houseType', "houseToward", "communityName", ]:
        lbl = LabelEncoder()
        data[col] = lbl.fit_transform(data[col])
    return data


data = get_feat(data)


# 一些特征工程
data['room_all'] = data['room_num'] + data['hall_num'] + data['toilet_num']
data['per_room'] = data['area'] / data['room_all']
# data['avg_money']=data['tradeMoney']/data['area']


# train,test分开
train=data[data.tradeMoney!=-1]
test=data[data.tradeMoney==-1]


# 只针对train的特征工程
train['per_area'] =  train['tradeMoney']/train['area']

# 只针对train的数据清洗
train=train[(train["per_area"]<1000)&(train["per_area"]>25) ]
train=train[(train["tradeMoney"]<18000)&(train["tradeMoney"]>100) ]
train=train[train["area"]<200]
train=train[train["area"]>10]
train = train [train['totalFloor'] <80]
train = train [train['tradeMeanPrice'] <100000]
train = train [train['remainNewNum'] <4000]

#查看train,test数据分布
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
sns.distplot(train['houseToward'].fillna(0),color="blue")
sns.distplot(test['houseToward'].fillna(0),color="red")


