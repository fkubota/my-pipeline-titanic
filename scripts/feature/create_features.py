import pandas as pd
import logging
from base import Feature, get_arguments, generate_features

# params
LOG_DIR = './../../logs/feature'
Feature.dir = './../../data/feature'


class FamilySize(Feature):
    def create_features(self):
        self.train['family_size'] = train['SibSp'] + train['Parch'] + 1
        self.test['family_size'] = test['SibSp'] + test['Parch'] + 1

class MyFeat1(Feature):
    def create_features(self):
        self.train['family_size'] = train['SibSp'] + train['Parch'] + 1
        self.test['family_size'] = test['SibSp'] + test['Parch'] + 1


if __name__ == '__main__':
    formatter = logging.Formatter('%(asctime)s %(name)s line %(lineno)d [%(levelname)s][%(funcName)s] %(message)s') # logger.setLevel('DEBUG')

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
    
    # do
    logger.info('-------------------- start')
    args = get_arguments()

    train = pd.read_csv('~/Git/my-pipeline-titanic/data/input/train.csv')
    test  = pd.read_csv('~/Git/my-pipeline-titanic/data/input/test.csv')
    generate_features(globals(), args.force)

    logger.info('-------------------- end')
