import pandas as pd
import logging
from base import Feature, get_arguments, generate_features
import base

# params
LOG_DIR = './../../logs/feature'
Feature.dir = './../../data/feature'


# logger
logging.basicConfig(level='DEBUG')
logger = logging.getLogger('create_features')


class FamilySize(Feature):
    def create_features(self):
        self.train['family_size'] = train['SibSp'] + train['Parch'] + 1
        self.test['family_size'] = test['SibSp'] + test['Parch'] + 1

class MyFeat1(Feature):
    def create_features(self):
        self.train['family_size'] = train['SibSp'] + train['Parch'] + 1
        self.test['family_size'] = test['SibSp'] + test['Parch'] + 1


if __name__ == '__main__':
    formatter = logging.Formatter('%(asctime)s %(name)s line %(lineno)d [%(levelname)s][%(funcName)s] %(message)s')
    logger.setLevel('DEBUG')

    # 標準出力
    sh = logging.StreamHandler()
    sh.setLevel('INFO')
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # ファイル出力
    fh = logging.FileHandler(f'{LOG_DIR}/log.log')
    fh.setLevel('DEBUG')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.debug('this is debugggg')

    # do
    logger.info('-------------------- start')

    args = get_arguments()

    train = pd.read_csv('~/Git/my-pipeline-titanic/data/input/train.csv')
    test  = pd.read_csv('~/Git/my-pipeline-titanic/data/input/test.csv')
    

    generate_features(globals(), args.force)

    logger.info('-------------------- end')
