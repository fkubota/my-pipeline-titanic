import sys
import warnings
from model_lgbm import ModelLGBM
from runner import Runner
import logging
from result import ResultHandler
# from util import Submission
sys.path.append('../utils')
from util import Util
from CONST import LOG_DIR
warnings.filterwarnings('ignore')


def preparation_logger(log_name):
    args = Util.get_arguments()
    form = '%(asctime)s %(name)s line%(lineno)d '\
           '[%(levelname)s][%(funcName)s] %(message)s'
    formatter = logging.Formatter(form)

    # logger 初期化
    logger = logging.getLogger('run')
    runner_logger = logging.getLogger('runner')
    util_logger = logging.getLogger('util')

    logger.setLevel(logging.DEBUG)
    runner_logger.setLevel(logging.DEBUG)
    util_logger.setLevel(logging.DEBUG)

    # 標準出力
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    # ハンドラー登録
    logger.addHandler(sh)
    runner_logger.addHandler(sh)
    util_logger.addHandler(sh)

    # debug?
    if args.debug:
        sh.setLevel(logging.DEBUG)
    else:
        # ファイル出力
        fh = logging.FileHandler(f'{LOG_DIR}/pipeline/{log_name}.log')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        runner_logger.addHandler(fh)
        util_logger.addHandler(fh)

    return logger, sh


def run(model, n_fold, feat_grps, model_params):
    rh = ResultHandler(model.__name__)
    print('==============', rh.name, '=======================================')
    logger, sh = preparation_logger(rh.name)
    logger.info('********************************************************')
    logger.info('******************** start pipeline ********************')
    logger.info('********************************************************')
    runner = Runner(model, n_fold, feat_grps, model_params, rh)
    runner.run_train_cv()
    runner.run_predict_cv()
    logger.info('********************************************************')
    logger.info('********************* end pipeline *********************')
    logger.info('********************************************************')


if __name__ == '__main__':

    # ================== set params ================================
    # run param
    n_fold = 2

    # 特徴量の指定
    feat_grps = ['FamilySize', 'Age_', 'Fare_', 'Sex_']

    # model_params
    model_params = {
     'n_estimators': 200,
     'boosting_type': 'gbdt',
     'max_depth': 5,
     'objective': 'binary',
     'num_leaves': 20,
     'learning_rate': 0.05,
     'max_bin': 512,
     'subsample_freq': 1,
     'colsample_bytree': 0.8,
     'reg_alpha': 5,
     'reg_lambda': 10,
     'min_child_weight': 1,
     'min_child_samples': 5,
     'metric': 'binary_logloss'
     }

    # ==============================================================

    # parse_params
    dict_params = model_params
    model = ModelLGBM
    dict_params_list = Util.parse_dict_param(dict_params)
    for params in dict_params_list:
        run(model=model,
            n_fold=n_fold,
            feat_grps=feat_grps,
            model_params=params)
