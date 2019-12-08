from abc import ABCMeta, abstractmethod
import os
import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import inspect
import logging
import pickle

# params
MEMO_PATH = './../../data/feature/_memo.csv'

# logger
logger = logging.getLogger('base')
logger.setLevel(logging.DEBUG)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', '-f', action='store_true',
                        help='Overwrite existing files')

    parser.add_argument('--test', '-t', action='store_true',
                        help='test mode')
    return parser.parse_args()


def get_features(namespace):
    for k, v in namespace.items():
        isclass = inspect.isclass(v)
        isabs = inspect.isabstract(v)
        if isclass and issubclass(v, Feature) and not isabs:
            yield v()


def generate_features(namespace, overwrite, istest):
    for f in get_features(namespace):
        path_tr_exist = f.feat_train_path.exists()
        path_te_exist = f.feat_test_path.exists()
        if path_tr_exist and path_te_exist and not overwrite:
            logger.info(f'skip {f.name}')
        else:
            f.run(overwrite).save(istest)


class Feature(metaclass=ABCMeta):
    prefix = ''
    suffix = ''
    dir = '.'

    def __init__(self):
        self.name = self.__class__.__name__
        self.now = datetime.datetime.now()
        self.feat_train = pd.DataFrame()
        self.feat_test = pd.DataFrame()
        self.feat_train_path = Path(self.dir) / f'{self.name}_train.pkl'
        self.feat_test_path = Path(self.dir) / f'{self.name}_test.pkl'
        self.basic_metas = np.array(['memo', 'type'])
        self.meta_dict = {'date': self.now}

    def run(self, overwrite):
        self.add_meta()
        self.check_meta()
        self.create_features()
        self.create_memo(overwrite)
        prefix = self.prefix + '_' if self.prefix else ''
        suffix = '_' + self.suffix if self.suffix else ''
        self.feat_train.columns = prefix + self.feat_train.columns + suffix
        self.feat_test.columns = prefix + self.feat_test.columns + suffix
        return self

    @abstractmethod
    def create_features(self):
        raise NotImplementedError

    @abstractmethod
    def add_meta(self, meta_dict):
        raise NotImplementedError

    def check_meta(self):
        keys_meta = self.meta_dict.keys()
        logic_check = [key in keys_meta for key in self.basic_metas]
        # basic_meta情報に抜けがないか確認
        if all(logic_check):
            logger.info('check_meta ... ok')
        else:
            idxs = np.logical_not(logic_check)
            text0 = f'{self.name} の basic meta のkeyが欠損しています。'
            text1 = 'deficiency_key: {self.basic_keys[idxs]}'
            raise RuntimeError(text0 + text1)

    def create_memo(self, overwrite):
        if os.path.isfile(MEMO_PATH):
            logger.debug('exists')
            df = pd.read_csv(MEMO_PATH)
        else:
            logger.debug('none')
            df = pd.DataFrame()
            df['date'] = self.now
            df['feat_name'] = [self.name]
            df['memo'] = [self.meta_dict['memo']]
            df.to_csv(MEMO_PATH, index=False, encoding='utf-8')
            return

        # overwrite オプション
        if overwrite:
            logger.info('overwrite memo')
            df = df[df['feat_name'] != self.name]

        # すでにメモに存在していたら、何もしない
        feat_names = df['feat_name'].values
        if self.name in feat_names:
            return

        _dict = {'date': [self.now],
                 'feat_name': [self.name],
                 'memo': [self.meta_dict['memo']]}
        new_df = pd.DataFrame(_dict)
        df = pd.concat([df, new_df], axis=0)
        logger.info(f'memo: {self.meta_dict["memo"]}')
        logger.debug(f'_memo.csv: {df}')
        df.to_csv(MEMO_PATH, index=False, encoding='utf-8')

    def save(self, istest):
        if istest:
            logger.info('not save feature')
            pass
        else:
            data = 0
            # with open(self.feat_train_path, mode='wb') as f:
            #     pickle.dump(data, f)
            # with open(self.feat_test_path, mode='wb') as f:
            #     pickle.dump(data, f)

        logger.debug(f'save path={self.feat_train_path}')
        logger.debug(f'save path={self.feat_test_path}')
        logger.debug(f'train feat size={self.feat_train.shape}')
        logger.debug(f'test  feat size={self.feat_test.shape}')
        logger.debug(f'train {self.feat_train.head()}')
        logger.debug(f'test {self.feat_test.head()}')
        logger.info(f' ===== finish {self.name} =====')
