from features import *
from GetData import *
from models import *
from Params import *
import pandas as pd
import time

import warnings
warnings.filterwarnings('ignore')

train, test, target, true_target, features, categorical_feats = getData(feature)

oof_lgb, predictions_lgb, feature_importance_df = lgb_model(params, train, test, target, true_target, features, categorical_feats,'lgb')
#oof_lgb, predictions_lgb, feature_importance_df = lgb_model2(params_new, train, test, target, true_target, features, categorical_feats,'lgb2')


tmp = test.copy()
tmp['trueArea'] = None
i = tmp['area'] <= 20
tmp['room_num'] = tmp['room_num'].astype('int')
tmp['hall_num'] = tmp['hall_num'].astype('int')
tmp['bathroom_num'] = tmp['bathroom_num'].astype('int')
tmp['trueArea'][i] = tmp['area'][i] * (tmp['room_num'][i] + tmp['hall_num'][i] + tmp['bathroom_num'][i]/3)
tmp['trueArea'][tmp['area'] > 20] = tmp['area'][tmp['area'] > 20]
tmp['tradeMoney'] = tmp['trueArea'] * predictions_lgb
test['tradeMoney'] = tmp['tradeMoney']
t = pd.read_csv('../data/test_a.csv')
test = test.reindex(t.index)
test['ID'] = test['ID'].astype('int64')
test = pd.merge(t, test, on=['ID'])

# t1 = pd.read_csv('../submit_05_10__23_32.csv',header=None)
# print(t1.corrwith(pd.DataFrame(test['tradeMoney'].values).astype('float'),method='pearson'))
#print(t1.corrwith(pd.DataFrame(predictions_lgb),method='pearson'))

name = "最终版"
test['tradeMoney'].apply(round).to_csv('../submit_version/'+name+'_' + time.strftime('%m_%d__%H_%M', time.localtime(time.time()))+'.csv', na_rep='\n', index=False, encoding='utf8', header=False)
#pd.DataFrame(predictions_lgb).apply(round).to_csv('submit/'+name+'_' + time.strftime('%m_%d__%H_%M', time.localtime(time.time()))+'.csv', na_rep='\n', index=False, encoding='utf8', header=False)

#基础特征－调参
#Predict Score: 0.89992 0.991496
#加入特征
#Predict Score: 0.90051 0.991552
#设置num_leaves=30
#Predict Score: 0.89998 0.99152
#加入两个队友的特征
#Predict Score: 0.90300 0.990084
#Predict Score: 0.90326 0.990028
#加入小区交易金额在板块中占比
#Predict Score: 0.90948 0.983575
    #加入小区室、厅、卫平均
    #Predict Score: 0.90941