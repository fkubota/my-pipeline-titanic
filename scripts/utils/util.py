import os
import itertools
import pickle
import json
import pandas as pd
import argparse
from logging import getLogger
from CONST import TRAIN_PATH, TEST_PATH
from CONST import FEAT_DIR, DEBUG_LENGTH, SUBMISSION_PATH

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
        # df = df[:DEBUG_LENGTH] if args.debug else df[:DEBUG_LENGTH]
        df = df[:DEBUG_LENGTH] if args.debug else df
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

    @classmethod
    def save_params(cls, data, save_dir):
        name = save_dir.split('/')[-1]
        path = f'{save_dir}/{name}_params.json'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        file_ = open(path, 'w')
        json.dump(data, file_, indent=4)

    @classmethod
    def save_oof(cls, data, save_dir):
        oof = cls.load_train_data()  # ['PassengerId', 'Survived']
        oof = oof[['PassengerId', 'Survived']]
        name = save_dir.split('/')[-1]
        path = f'{save_dir}/{name}_oof.csv'
        oof['Survived'] = data
        oof.to_csv(path, index=False, encoding='utf-8')

    @classmethod
    def save_submission(cls, data, save_dir):
        submission = pd.read_csv(SUBMISSION_PATH)
        submission['Survived'] = data
        name = save_dir.split('/')[-1]
        path = f'{save_dir}/{name}_submission.csv'
        submission.to_csv(path, index=False, encoding='utf-8')

    @classmethod
    def parse_dict_param(cls, dict_param):
        keys = list(dict_param.keys())

        # param がリストの場合、その分だけグリッドを作る
        params_list = []
        for key in keys:
            val = [dict_param[key]] if type(dict_param[key]) \
                                    != list else dict_param[key]
            params_list.append(val)

        p = itertools.product(*params_list)
        dict_param_list = []
        for a in p:
            param_dict = {}
            for i, val in enumerate(a):
                param_dict[keys[i]] = val
            dict_param_list.append(param_dict)
        return dict_param_list


if __name__ == '__main__':
    # print(load_train_data().head())
    # print(load_test_data().head())
    # print(Util.load_test_features(['FamilySize', 'Title']))
    a = Util.load_train_data()
    print(a.shape)
