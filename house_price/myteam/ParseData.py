import numpy as np
import pandas as pd

dropID = [100200482, 100151139, 100143182, 100134362, 100107357, 100105753, 100104337, 100097746, 100097735, 100067761, 100050443, 100050179, 100034539, 100018725, 100015803, 100006576, 100006321, 100000320, 100150584, 100131099, 100025094, 100018308, 100305713, 100124232, 100311050, 100308688, 100305658, 100256667, 100236239, 100107392, 100102076, 100098157, 100093190, 100089652, 100088555, 100064787, 100028461, 100024047, 100018144, 100313563, 100313523, 100313154, 100313105, 100313095, 100313080, 100312344, 100311414, 100311320, 100311059, 100308204, 100307649, 100307200, 100307075, 100306274, 100306225, 100269778, 100262508, 100250321, 100240685, 100230239, 100229103, 100216016, 100207373, 100206183, 100203987, 100191454, 100186272, 100179265, 100149233, 100148508, 100141120, 100139430, 100135293, 100134216, 100132619, 100131218, 100129795, 100128505, 100125389, 100119157, 100119124, 100116746, 100115097, 100114447, 100112345, 100109091, 100108480, 100107705, 100107551, 100107511, 100106168, 100102808, 100102075, 100102073, 100101422, 100100923, 100100012, 100096841, 100095652, 100095591, 100095327, 100094841, 100091975, 100090924, 100090570, 100088687, 100088633, 100088463, 100088246, 100087591, 100086528, 100085496, 100084647, 100083356, 100083355, 100079293, 100078958, 100077356, 100077201, 100076165, 100075875, 100074627, 100074470, 100073694, 100073615, 100072654, 100072641, 100068719, 100067997, 100064456, 100064160, 100064075, 100062757, 100062735, 100062454, 100060170, 100060167, 100058917, 100056736, 100056477, 100054925, 100054474, 100052076, 100048412, 100048320, 100044314, 100044034, 100044021, 100042739, 100042174, 100040171, 100036712, 100036704, 100036119, 100035393, 100034187, 100033810, 100032709, 100032310, 100030314, 100030221, 100028770, 100028167, 100026320, 100026109, 100026086, 100026050, 100024714, 100024108, 100021370, 100019812, 100018841, 100018042, 100017810, 100016314, 100014313, 100012716, 100012314, 100012137, 100010669, 100009428, 100008149, 100006109, 100006037, 100005812, 100005096, 100004135, 100004012, 100003804, 100002712, 100001807, 100000312, 100082629, 100130478, 100127915, 100112385, 100096460, 100094801, 100078752, 100048053, 100036006, 100312203, 100312005, 100096615, 100018548, 100017369, 100124493, 100118780, 100109397]


# 减内存占用
def reduce_mem_usage(df, verbose=True):
    """
    减少内存
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
                start_mem - end_mem) / start_mem))
    return df


# 数据预处理
def parseData(df):
    """
    数据预处理
    """
    # 将houseType转化为‘房间数’，‘厅数’，‘卫生间数’
    def parseRoom(info, index):
        if index == 0:
            try:
                res = int(info.split('室')[0])
            except Exception as e:
                print(e)
                res = 0

        elif index == 1:
            try:
                res = int(info.split('室')[1].split('厅')[0])
            except Exception as e:
                print(e)
                res = 0

        elif index == 2:
            try:
                res = int(info.split('室')[1].split('厅')[1].split('卫')[0])
            except Exception as e:
                print(e)
                res = 0

        return res
    df.insert(3, 'room_num', None)
    df.insert(4, 'hall_num', None)
    df.insert(5, 'bathroom_num', None)
    df['room_num'] = df['houseType'].apply(lambda x: parseRoom(x, 0))
    df['hall_num'] = df['houseType'].apply(lambda x: parseRoom(x, 1))
    df['bathroom_num'] = df['houseType'].apply(lambda x: parseRoom(x, 2))
    df.drop('houseType', axis=1, inplace=True)

    df['rentType'][df['rentType'] == '--'] = '未知方式'
    # 转换object类型数据
    columns = ['ID', 'rentType', 'houseFloor', 'houseToward', 'houseDecoration', 'communityName', 'region', 'plate']
    for col in columns:
        df[col] = df[col].astype('category')

    # 去掉无用列
    drop_columns = ['supplyLandNum', 'supplyLandArea', 'tradeLandNum', 'tradeLandArea', 'landTotalPrice',
                    'landMeanPrice']
    df.drop('city', axis=1, inplace=True)
    # df.drop(drop_columns, axis=1, inplace=True)

    # 将buildYear列转换为整型数据
    tmp = df['buildYear'].copy()
    tmp2 = tmp[tmp != '暂无信息'].astype('int')
    # 将没有修建年份的数据用众数替代
    tmp[tmp == '暂无信息'] = tmp2.mode().iloc[0]
    df['buildYear'] = tmp
    df['buildYear'] = df['buildYear'].astype('int')

    # 处理pv和uv的空值
    pv_fill_value = 528
    uv_fill_value = 141
    df['pv'].fillna(pv_fill_value, inplace=True)
    df['uv'].fillna(uv_fill_value, inplace=True)
    df['pv'] = df['pv'].astype('int')
    df['uv'] = df['uv'].astype('int')

    return df


# 数据清洗
def washDF(df_train, df_test):
    """
    清洗数据
    """
    # 　除去训练集中离群值
    df_train = df_train[df_train['area'] <= 700]  # 700
    df_train = df_train[df_train['tradeMoney'] <= 20000]  # 100000
    #df_train = df_train[df_train['totalFloor'] <= 40]
    df_train = df_train[df_train['area'] >= 7]
    df_train = df_train[df_train['tradeMoney'] > 10]
    #df_train = df_train[df_train['saleSecHouseNum'] < 25]  # 14
    #df_train = df_train[df_train['lookNum'] < 20]
    #df_train = df_train[df_train['ID'].apply(lambda x: x not in dropID)] #预测偏差大于5000的数据

    second = df_train['totalWorkers'][df_train['totalWorkers'] != 855400].max() + 100000
    df_train['totalWorkers'][df_train['totalWorkers'] == 855400] = second
    df_test['totalWorkers'][df_test['totalWorkers'] == 855400] = second

    plate_list = ['BK00032', 'BK00001']
    for plate in plate_list:
        df_train = df_train[df_train['plate'] != plate]

    df_train = df_train[df_train['region'] != 'RG00015']

    return df_train, df_test


# 数据归一化
def min_max_scaler(data):
    """
    数据归一化
    """
    data = (data - data.min()) / (data.max() - data.min())
    return data


#高档小区
def classifyCommunity(df):
    # 选出高档小区,用于得到features.py中的clist
    community_list = list(set(df['communityName'].values))
    mean_tradeM = list(range(len(community_list)))
    for i, item in enumerate(community_list):
        mean_tradeM[i] = df['tradeMoney'][df['communityName'] == item].mean()
    mean_tradeM = pd.DataFrame(mean_tradeM)
    mean_tradeM.fillna(mean_tradeM.mean(), inplace=True)
    p = np.percentile(mean_tradeM, (10, 40, 60, 75), interpolation='midpoint')
    list1 = list(pd.DataFrame(community_list)[mean_tradeM > p[3]].dropna()[0])
    return list1