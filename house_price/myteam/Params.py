params = {
    'num_leaves': 33,#31
    'min_data_in_leaf': 25,#15
    'objective': 'regression',
    #'max_depth': 8,#9
    'learning_rate': 0.01,# 0.01
    'min_child_weight': 0,
    "feature_fraction": 0.7,
    "bagging_freq": 2,
    "bagging_fraction": 0.8,#0.9
    "min_split_gain": 0.1,#0.01
    "metric": 'r2_metric', #r2_metric
    "bagging_seed": 11,
    "lambda_l1": 0.32, #0.2
    'lambda_l2':10,#6
    'cat_smooth': 5,
    "verbosity": -1,
    "nthread": 4,
}


params2 = {
    'num_leaves': 31,
    'min_child_samples':20,
    'objective': 'regression',
    'learning_rate': 0.01,
    "boosting": "gbdt",
    "feature_fraction": 0.8,
    "bagging_freq": 1,
    "bagging_fraction": 0.85,
    "bagging_seed": 23,
    "metric": 'rmse',
    "lambda_l1": 0.2,
    "nthread": 4,
}


params_new = {
    'num_leaves': 30,#50
    'min_data_in_leaf': 25,
    'objective': 'regression',
    'max_depth': 6,
    'learning_rate': 0.01,
    'min_child_weight': 0.363,
    "feature_fraction": 0.7903,
    "bagging_freq": 3,
    "bagging_fraction": 0.7897,
    "min_split_gain": 0.81,
    "metric": 'r2_metric',
    "bagging_seed": 11,
    "lambda_l1": 0.5328,
    'lambda_l2': 1.703,
    'cat_smooth': 2.204,
    "verbosity": -1,
    "nthread": 4,
}