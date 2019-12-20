import sys
from model_xgb import ModelXGB
from runner import Runner
import logging
from result import ResultHandler
# from util import Submission
sys.path.append('../utils')
from util import Util
from CONST import LOG_DIR


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
    feat_grps = ['FamilySize']

    # xgb pram
    params_xgb = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'num_class': 1,
        'max_depth': [100, 200, 300],
        'eta': 0.01,
        'min_child_weight': 10,
        'subsample': 0.9,
        'colsample_bytree': 0.8,
        # 'silent': 1,
        'random_state': 71,
        'verbosity': 2,
        'verbose': 10,
        'num_round': 10,
        'early_stopping_rounds': 100,
    }

    # ==============================================================

    # parse_params
    dict_params = params_xgb
    model = ModelXGB
    dict_params_list = Util.parse_dict_param(dict_params)
    for params in dict_params_list:
        run(model=model,
            n_fold=n_fold,
            feat_grps=feat_grps,
            model_params=params)
    # Submission.create_submission('xgb1')
