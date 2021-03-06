import sys
import logging
from base import Feature, generate_features
sys.path.append('../utils')
from CONST import LOG_DIR
from util import Util


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
    fh = logging.FileHandler(f'{LOG_DIR}/feature/log.log')
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


class Fare_(Feature):
    def create_features(self):
        self.feat_train['Fare_'] = train['Fare'].fillna(train['Fare'].mean())
        self.feat_test['Fare_'] = test['Fare'].fillna(test['Fare'].mean())

    def add_meta(self):
        self.meta_dict['memo'] = 'fare, 欠損値は、平均で埋めた'
        self.meta_dict['num_or_cat'] = 'num'
        self.meta_dict['date'] = '{0:%Y-%m-%d %H:%M:%S}'.format(self.now)


class Age_(Feature):
    def create_features(self):
        self.feat_train['Age_'] = train['Age'].fillna(train['Age'].mean())
        self.feat_test['Age_'] = test['Age'].fillna(test['Age'].mean())

    def add_meta(self):
        self.meta_dict['memo'] = 'age, 欠損値は、平均で埋めた'
        self.meta_dict['num_or_cat'] = 'num'
        self.meta_dict['date'] = '{0:%Y-%m-%d %H:%M:%S}'.format(self.now)


class Sex_(Feature):
    def create_features(self):
        self.feat_train['Sex_'] = train['Sex'].apply(lambda x: 1 if x == 'male' else 0)
        self.feat_test['Sex_'] = test['Sex'].apply(lambda x: 1 if x == 'male' else 0)

    def add_meta(self):
        self.meta_dict['memo'] = 'sex'
        self.meta_dict['num_or_cat'] = 'cat'
        self.meta_dict['date'] = '{0:%Y-%m-%d %H:%M:%S}'.format(self.now)


if __name__ == '__main__':
    # log
    logger, sh, fh = preparation_logger()

    # do
    args = Util.get_arguments()
    train = Util.load_train_data()
    test = Util.load_test_data()

    # test mode?
    if args.debug:
        fh.setLevel(logging.ERROR)  # file書き出ししないという意思表示
        sh.setLevel(logging.DEBUG)  # stream handler を infoからdebugへ
        logger.info('*******************************')
        logger.info('********** test mode **********')
        logger.info('*******************************')

    logger.info('-------------------- start')
    logger.debug(f'\n-train\n {train.head()}')
    logger.debug(f'\n-test\n {test.head()}')
    generate_features(globals(), args.force, args.debug)
    logger.info('-------------------- end')
