import os
import pickle
import pandas as pd
import argparse
from logging import getLogger
from CONST import TRAIN_PATH, TEST_PATH, FEAT_DIR, DEBUG_LENGTH

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
        args = cls.get_arguments()
        df = pd.read_csv(path)
        df = df[:DEBUG_LENGTH] if args.debug else df[:DEBUG_LENGTH]
        return df

    @classmethod
    def load_train_data(cls):
        df = cls.load_csv(TRAIN_PATH)
        return df

    @classmethod
    def load_test_data(cls):
        df = cls.load_csv(TEST_PATH)
        return df

    @classmethod
    def load_train_features(cls, feat_grps):
        df = [cls.load(f'{FEAT_DIR}/{fg}_train.pkl') for fg in feat_grps]
        df = pd.concat(df, axis=1)
        return df

    @classmethod
    def load_test_features(cls, feat_grps):
        df = [cls.load(f'{FEAT_DIR}/{fg}_test.pkl') for fg in feat_grps]
        df = pd.concat(df, axis=1)
        return df

    @classmethod
    def get_arguments(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument('--force', '-f', action='store_true',
                            help='Overwrite existing files')

        parser.add_argument('--debug', '-d', action='store_true',
                            help='debug mode')
        return parser.parse_args()


if __name__ == '__main__':
    # print(load_train_data().head())
    # print(load_test_data().head())
    # print(Util.load_test_features(['FamilySize', 'Title']))
    a = Util.load_train_data()
    print(a.shape)
