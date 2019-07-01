from bayes_opt import BayesianOptimization
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import lightgbm as lgb
import gc
from GetData import getData
from features import feature
from models import r2_metric
import warnings
warnings.filterwarnings('ignore')

train_, _, _, true_target_, features, categorical_feats = getData(feature)
bayesian_tr_index, bayesian_val_index = list(StratifiedKFold(n_splits=2, shuffle=True, random_state=1).split(train_.values, true_target_.values))[0]

train = train_.iloc[bayesian_tr_index].copy()
true_target = true_target_.iloc[bayesian_tr_index].copy()
test = train_.iloc[bayesian_val_index].copy()
test_target = true_target_.iloc[bayesian_val_index].copy()

def LGB_bayesian(num_leaves,
                 min_data_in_leaf,
                 max_depth,
                 min_child_weight,
                 feature_fraction,
                 bagging_freq,
                 bagging_fraction,
                 min_split_gain,
                 lambda_l1,
                 lambda_l2,cat_smooth):
    num_leaves = int(num_leaves)
    max_depth = int(max_depth)
    bagging_freq = int(bagging_freq)
    min_data_in_leaf = int(min_data_in_leaf)

    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    oof_lgb = np.zeros(len(train))
    predictions_lgb = np.zeros(len(test))

    for fold_, (trn_idx, val_idx) in enumerate(kf.split(train.values, true_target.values)):
        print("fold n°{}".format(fold_))
        trn_data = lgb.Dataset(train[features].iloc[trn_idx], label=true_target.iloc[trn_idx],
                               categorical_feature=categorical_feats)
        val_data = lgb.Dataset(train[features].iloc[val_idx], label=true_target.iloc[val_idx],
                               categorical_feature=categorical_feats)

        num_round = 30000

        params = {
            'num_leaves': num_leaves,
            'min_data_in_leaf': min_data_in_leaf,
            'objective': 'regression',
            'max_depth': max_depth,
            'learning_rate': 0.01,
            'min_child_weight': min_child_weight,
            "feature_fraction": feature_fraction,
            "bagging_freq": bagging_freq,
            "bagging_fraction": bagging_fraction,
            "min_split_gain": min_split_gain,
            "metric": 'r2_metric',
            "bagging_seed": 11,
            "lambda_l1": lambda_l1,
            'lambda_l2':lambda_l2,
            'cat_smooth': cat_smooth,
            "verbosity": -1,
            "nthread": 4,
        }
        clf = lgb.train(params, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=500,
                        early_stopping_rounds=200, feval=r2_metric)
        oof_lgb[val_idx] = clf.predict(train[features].iloc[val_idx], num_iteration=clf.best_iteration)
        predictions_lgb += clf.predict(test[features], num_iteration=clf.best_iteration) / kf.n_splits
        gc.collect()
    print("Train:",r2_score(true_target, oof_lgb))
    return r2_score(test_target, predictions_lgb)


bounds_LGB = {
    'num_leaves': (5, 40),
    'min_data_in_leaf': (10, 25),
    'max_depth': (3, 15),
    'min_child_weight': (0.0, 1.0),
    "feature_fraction": (0.5, 0.9),
    "bagging_freq": (1, 50),
    "bagging_fraction": (0.5, 0.9),
    "min_split_gain": (0.0, 1.0),
    "lambda_l1": (0.0, 5.0),
    'lambda_l2': (0.0, 10.0),
    'cat_smooth': (0, 20),
}


LGB_BO = BayesianOptimization(LGB_bayesian, bounds_LGB, random_state=16)

LGB_BO.maximize()

print(LGB_BO.max['target'])
print(LGB_BO.max['params'])