import os
import pickle
import pandas as pd
from logging import getLogger
from CONST import TRAIN_PATH, TEST_PATH, FEAT_DIR

logger = getLogger('util')


class Util:

    @classmethod
    def save(cls, data, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, mode='wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path, mode='rb') as f:
            data = pickle.load(f)
        return data

    @classmethod
    def load_csv(cls, path):
        logger.debug('enter')
        df = pd.read_csv(path)
        logger.debug('exit')
        return df

    @classmethod
    def load_train_data(cls):
        logger.debug('enter')
        df = cls.load_csv(TRAIN_PATH)
        logger.debug('exit')
        return df

    @classmethod
    def load_test_data(cls):
        logger.debug('enter')
        df = cls.load_csv(TEST_PATH)
        logger.debug('exit')
        return df

    @classmethod
    def load_train_features(cls, feat_grps):
        df = [cls.load(f'{FEAT_DIR}/{fg}_train.pkl') for fg in feat_grps]
        df = pd.concat(df, axis=1)
        return df

    @classmethod
    def load_test_features(cls, feat_grps):
        a = [print(b) for b in feat_grps]
        df = [cls.load(f'{FEAT_DIR}/{fg}_test.pkl') for fg in feat_grps]
        df = pd.concat(df, axis=1)
        return df


if __name__ == '__main__':
    # print(load_train_data().head())
    # print(load_test_data().head())
    print(Util.load_test_features(['FamilySize', 'Title']))
