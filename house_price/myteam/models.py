from sklearn.metrics import r2_score
from Params import *
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from ParseData import *
import gc


# 评分函数
def r2_metric(target, data):
    """
    自定义评分函数(赛题评分标准)
    """
    return 'score', r2_score(target, data.get_label()), True


# lgb单模型
def lgb_model(params, train, test, target, true_target, features, categorical_feats, name):
    tmp = train.copy()
    tmp['trueArea'] = None
    i = tmp['area'] <= 20
    tmp['trueArea'][i] = tmp['area'][i] * (tmp['room_num'][i] + tmp['hall_num'][i] + tmp['bathroom_num'][i] / 3)
    tmp['trueArea'][tmp['area'] > 20] = tmp['area'][tmp['area'] > 20]

    folds = KFold(n_splits=5, shuffle=True, random_state=1949)  # 2019
    oof_lgb = np.zeros(len(train))
    predictions_lgb = np.zeros(len(test))
    feature_importance_df = pd.DataFrame()

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
        print("fold n°{}".format(fold_))
        trn_data = lgb.Dataset(train[features].iloc[trn_idx], label=target.iloc[trn_idx],
                               categorical_feature=categorical_feats)
        val_data = lgb.Dataset(train[features].iloc[val_idx], label=target.iloc[val_idx],
                               categorical_feature=categorical_feats)

        num_round = 30000
        clf = lgb.train(params, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=500,
                        early_stopping_rounds=50, feval=r2_metric)
        oof_lgb[val_idx] = clf.predict(train[features].iloc[val_idx], num_iteration=clf.best_iteration)

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = features
        fold_importance_df["importance"] = clf.feature_importance()
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        predictions_lgb += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits
        gc.collect()

    tmp['tradeMoney'] = tmp['trueArea'] * oof_lgb
    print("CV Score: {:<8.5f}".format(1 - sum((oof_lgb - target) ** 2) / sum((target - target.mean()) ** 2)))
    print("Predict Score: {:<8.5f}".format(r2_score(true_target, tmp['tradeMoney'])))
    return oof_lgb, predictions_lgb, feature_importance_df


def lgb_model2(params, train, test, target, true_target, features, categorical_feats, name):
    folds = KFold(n_splits=5, shuffle=True, random_state=1949)  # 2019
    oof_lgb = np.zeros(len(train))
    predictions_lgb = np.zeros(len(test))
    feature_importance_df = pd.DataFrame()

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, true_target.values)):
        print("fold n°{}".format(fold_))
        trn_data = lgb.Dataset(train[features].iloc[trn_idx], label=true_target.iloc[trn_idx],
                               categorical_feature=categorical_feats)
        val_data = lgb.Dataset(train[features].iloc[val_idx], label=true_target.iloc[val_idx],
                               categorical_feature=categorical_feats)

        num_round = 30000
        clf = lgb.train(params, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=500,
                        early_stopping_rounds=50, feval=r2_metric)
        oof_lgb[val_idx] = clf.predict(train[features].iloc[val_idx], num_iteration=clf.best_iteration)

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = features
        fold_importance_df["importance"] = clf.feature_importance()
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        predictions_lgb += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits
        gc.collect()

    print("Predict Score: {:<8.5f}".format(r2_score(true_target, oof_lgb)))
    return oof_lgb, predictions_lgb, feature_importance_df
