import pandas as pd
from ParseData import *
from features import *


def getData(feature):
    train = pd.read_csv('../data/train_data.csv')
    test = pd.read_csv('../data/test_a.csv')

    # 预处理
    train = parseData(train)
    test = parseData(test)
    # 清洗
    train, test = washDF(train, test)

    # 避免在做特征的过程中顺序被打乱
    tmp1 = pd.DataFrame(train.copy()['ID'])
    tmp2 = pd.DataFrame(test.copy()['ID'])

    train, test = getMeanFloor(train, test)
    train, test = getSecondHandRatio(train, test)
    train, test = getTradeMoneyRatio(train, test)
    # train, test = getPlateMeanRoom(train, test)

    getNumOfFeat(train, test, 'communityName')
    getNumOfFeat(train, test, 'plate')
    getNumOfFeat(train, test, 'region')

    train, col = feature(train)
    test, col = feature(test)

    print("Getting lastMonthTradeMeanPrice...", end='\t')
    train, test = getLastMonthTradeMeanPrice(train, test)
    train['二手房交易趋势'] = train['tradeMeanPrice'] - train.pop('lastMonthTradeMeanPrice')
    test['二手房交易趋势'] = test['tradeMeanPrice'] - test.pop('lastMonthTradeMeanPrice')
    print("done\n")
    # """

    # 将两个类别特征转化为其每个类别对应出现的次数
    countCols = [
        'houseToward', 'houseDecoration',
    ]
    for c in countCols:
        encode_feature(train, test, c, col)

    # 还原数据顺序及数据类型
    train = pd.merge(tmp1, train, on=['ID'])
    test = pd.merge(tmp2, test, on=['ID'])
    train['region'] = train['region'].astype('category')
    train['plate'] = train['plate'].astype('category')
    test['region'] = test['region'].astype('category')
    test['plate'] = test['plate'].astype('category')

    # 重新清洗特征
    # train = train[train['房间面积'] <= 40]
    train = train[train['totalFloor'] > 1]

    true_target = train.pop('tradeMoney')
    tmp = train.copy()
    tmp['trueArea'] = None
    i = tmp['area'] <= 20
    tmp['trueArea'][i] = tmp['area'][i] * (tmp['room_num'][i] + tmp['hall_num'][i] + tmp['bathroom_num'][i] / 3)
    tmp['trueArea'][tmp['area'] > 20] = tmp['area'][tmp['area'] > 20]
    train_target = true_target / tmp['trueArea']

    features = [col for col in train.columns if col not in ['ID', ]]
    categorical_feats = col

    # classifyByKMeans(train, test, features, categorical_feats, features, 'CLASS', 3)
    # categorical_feats.append("CLASS")
    # 利用房屋相关的特征对房屋本身进行无监督聚类作为一个新特征

    feats1 = ['rentType', 'bathroom_num', 'hall_num', 'room_num', 'houseFloor', 'totalFloor']
    classifyByKMeans(train, test, features, categorical_feats, feats1, '房型', 4)
    categorical_feats.append("房型")
    """
    feats2 = ['subwayStationNum','busStationNum','interSchoolNum','schoolNum','privateSchoolNum','hospitalNum','drugStoreNum','gymNum','bankNum','shopNum','parkNum','mallNum','superMarketNum']
    classifyByKMeans(train, test, features, categorical_feats, feats2, '板块类别', 5)
    categorical_feats.append("板块类别")
    """
    train = reduce_mem_usage(train)
    test = reduce_mem_usage(test)

    return train, test, train_target, true_target, features, categorical_feats