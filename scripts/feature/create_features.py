import datetime
import pandas as pd
import logging
from base import Feature, get_arguments, generate_features

# params
LOG_DIR = './../../logs/feature'
Feature.dir = './../../data/feature'
LENGTH = 100


def preparation_logger():
    form = '%(asctime)s %(name)s line %(lineno)d [%(levelname)s][%(funcName)s] %(message)s'
    formatter = logging.Formatter(form)

    logger = logging.getLogger('create_features')
    logger.setLevel(logging.DEBUG)
    base_logger = logging.getLogger('base')
    base_logger.setLevel(logging.DEBUG)

    # 標準出力
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    # ファイル出力
    fh = logging.FileHandler(f'{LOG_DIR}/log.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # ハンドラー登録
    logger.addHandler(sh)
    logger.addHandler(fh)
    base_logger.addHandler(sh)
    base_logger.addHandler(fh)
    return logger, sh, fh


class FamilySize(Feature):
    def create_features(self):
        self.feat_train['family_size'] = train['SibSp'] + train['Parch'] + 1
        self.feat_test['family_size'] = test['SibSp'] + test['Parch'] + 1

    def add_meta(self):
        now = datetime.datetime.now()
        self.meta_dict['type'] = 'int'
        self.meta_dict['date'] = '{0:%Y-%m-%d %H:%M:%S}'.format(now)


class MyFeat1(Feature):
    def create_features(self):
        self.feat_train['family_size'] = train['SibSp'] + train['Parch'] + 1
        self.feat_test['family_size'] = test['SibSp'] + test['Parch'] + 1

    def add_meta(self):
        now = datetime.datetime.now()
        self.meta_dict['type'] = 'int'
        self.meta_dict['date'] = '{0:%Y-%m-%d %H:%M:%S}'.format(now)


if __name__ == '__main__':
    # log
    logger, _, fh = preparation_logger()

    # do
    args = get_arguments()

    train = pd.read_csv('~/Git/my-pipeline-titanic/data/input/train.csv')
    test = pd.read_csv('~/Git/my-pipeline-titanic/data/input/test.csv')

    # test mode?
    if args.test:
        fh.setLevel(logging.ERROR)  # file書き出ししないという意思表示
        logger.info('********** test mode **********')
        train = train[:LENGTH]
        test = test[:LENGTH]

    logger.info('-------------------- start')
    generate_features(globals(), args.force, args.test)
    logger.info('-------------------- end')
