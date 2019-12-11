import numpy as np
import pandas as pd

# From model_nn import ModelNN
from model_xgb import ModelXGB
from runner import Runner
import logging
# from util import Submission

def preparation_logger():
    logger = logging.getLogger('run')
    runner_logger = logging.getLogger('runner')
    logger.setLevel(logging.DEBUG)
    runner_logger.setLevel(logging.DEBUG)
    
    # 標準出力
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)

    # ハンドラー登録
    logger.addHandler(sh)
    runner_logger.addHandler(sh)

    return logger, sh


if __name__ == '__main__':
    # logging
    logger, sh = preparation_logger()
    logger.info('******************** start pipeline ********************')


    params_xgb = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'num_class': 1,
        'max_depth': 12,
        'eta': 0.1,
        'min_child_weight': 10,
        'subsample': 0.9,
        'colsample_bytree': 0.8,
        'silent': 1,
        'random_state': 71,
        'num_round': 10,
        'early_stopping_rounds': 10,
    }

    params_xgb_all = dict(params_xgb)
    params_xgb_all['num_round'] = 30 # 350

    # 特徴量の指定
    features = ['Age']

    # xgboostによる学習・予測
    runner = Runner('xgb1', ModelXGB, features, params_xgb)
    runner.run_train_cv()
    runner.run_predict_cv()
    logger.info('******************** end pipeline ********************')
    # Submission.create_submission('xgb1')