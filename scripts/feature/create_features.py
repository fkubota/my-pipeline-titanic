import sys
import pandas as pd
import logging
from base import Feature, get_arguments, generate_features
sys.path.append('../../utils')
from CONST import LOG_DIR, LENGTH

# params
Feature.dir = './../../data/feature'


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
        self.meta_dict['memo'] = 'n_person in family'
        self.meta_dict['num_or_cat'] = 'num'
        self.meta_dict['date'] = '{0:%Y-%m-%d %H:%M:%S}'.format(self.now)


class Title(Feature):
    def create_features(self):
        self.feat_train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.',
                                                             expand=False)
        self.feat_test['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.',
                                                            expand=False)

    def add_meta(self):
        self.meta_dict['memo'] = '敬称(Ms, Masterとか)'
        self.meta_dict['num_or_cat'] = 'cat'
        self.meta_dict['date'] = '{0:%Y-%m-%d %H:%M:%S}'.format(self.now)


if __name__ == '__main__':
    # log
    logger, sh, fh = preparation_logger()

    # do
    args = get_arguments()

    train = pd.read_csv('~/Git/my-pipeline-titanic/data/input/train.csv')
    test = pd.read_csv('~/Git/my-pipeline-titanic/data/input/test.csv')

    # test mode?
    if args.test:
        fh.setLevel(logging.ERROR)  # file書き出ししないという意思表示
        sh.setLevel(logging.DEBUG)  # stream handler を infoからdebugへ
        logger.info('********** test mode **********')
        train = train[:LENGTH]
        test = test[:LENGTH]

    logger.info('-------------------- start')
    logger.debug(f'\n-train\n {train.head()}')
    logger.debug(f'\n-test\n {test.head()}')
    generate_features(globals(), args.force, args.test)
    logger.info('-------------------- end')
