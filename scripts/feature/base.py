from abc import ABCMeta, abstractmethod
from pathlib import Path
import pandas as pd
import argparse
import inspect
import logging
import pickle

# logger
logger = logging.getLogger('base')
logger.setLevel(logging.DEBUG)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', '-f', action='store_true',
                        help='Overwrite existing files')
    return parser.parse_args()


def get_features(namespace):
    for k, v in namespace.items():
        isclass = inspect.isclass(v)
        isabs = inspect.isabstract(v)
        if isclass and issubclass(v, Feature) and not isabs:
            yield v()


def generate_features(namespace, overwrite):
    for f in get_features(namespace):
        if f.feat_train_path.exists() and f.feat_test_path.exists() and not overwrite:
            print(f.name, 'was skipped')
            logger.info(f'skip {f.name}')
        else:
            f.run().save()


class Feature(metaclass=ABCMeta):
    prefix = ''
    suffix = ''
    dir = '.'

    def __init__(self):
        self.name = self.__class__.__name__
        self.feat_train = pd.DataFrame()
        self.feat_test = pd.DataFrame()
        self.feat_train_path = Path(self.dir) / f'{self.name}_train.pkl'
        self.feat_test_path = Path(self.dir) / f'{self.name}_test.pkl'

    def run(self):
        self.create_features()
        prefix = self.prefix + '_' if self.prefix else ''
        suffix = '_' + self.suffix if self.suffix else ''
        self.feat_train.columns = prefix + self.feat_train.columns + suffix
        self.feat_test.columns = prefix + self.feat_test.columns + suffix
        return self

    @abstractmethod
    def create_features(self):
        raise NotImplementedError

    def add_meta(self):
        feat_train_dict = {}
        feat_test_dict = {}

        feat_train_dict['df'] = self.feat_train
        feat_test_dict['df'] = self.feat_test

    def save(self):
        data = 0
        with open(self.feat_train_path, mode='wb') as f:
            pickle.dump(data, f)

        logger.debug(f'save path={self.feat_train_path}')
        logger.debug(f'save path={self.feat_test_path}')
        logger.debug(f'train feat size={self.feat_train.shape}')
        logger.debug(f'test  feat size={self.feat_test.shape}')
        logger.debug(f'train {self.feat_train.head()}')
        logger.debug(f'test {self.feat_test.head()}')
        logger.info(f' ===== finish {self.name} =====')

