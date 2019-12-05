import re
import time
from abc import ABCMeta, abstractmethod
from pathlib import Path
from contextlib import contextmanager
import pandas as pd
import argparse
import inspect
import logging

# logger
# logger = logging.getLogger(__name__)
logger = logging.getLogger('create_features').getChild('base')
logger.setLevel('DEBUG')
# formatter = logging.Formatter('%(asctime)s %(name)s line %(lineno)d [%(levelname)s][%(funcName)s] %(message)s')

# 標準出力
# sh = logging.StreamHandler()
# sh.setLevel('INFO')
# sh.setFormatter(formatter)
# logger.addHandler(sh)
# 
# # ファイル出力
# fh = logging.FileHandler(f'{LOG_DIR}/log.log')
# fh.setLevel('DEBUG')
# fh.setFormatter(formatter)
# logger.addHandler(fh)




def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', '-f', action='store_true', help='Overwrite existing files')
    return parser.parse_args()


def get_features(namespace):
    for k, v in namespace.items():
        print(k, v)
        if inspect.isclass(v) and issubclass(v, Feature) and not inspect.isabstract(v):
            yield v()


def generate_features(namespace, overwrite):
    for f in get_features(namespace):
        if f.train_path.exists() and f.test_path.exists() and not overwrite:
            print(f.name, 'was skipped')
        else:
            f.run().save()

@contextmanager
def timer(name):
    t0 = time.time()
    # print(f'[{name}] start')
    yield
    # print(f'[{name}] done in {time.time() - t0:.0f} s')


class Feature(metaclass=ABCMeta):
    prefix = ''
    suffix = ''
    dir = '.'
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.train_path = Path(self.dir) / f'{self.name}_train.pkl'
        self.test_path = Path(self.dir) / f'{self.name}_test.pkl'
    
    def run(self):
        with timer(self.name):
            self.create_features()
            prefix = self.prefix + '_' if self.prefix else ''
            suffix = '_' + self.suffix if self.suffix else ''
            self.train.columns = prefix + self.train.columns + suffix
            self.test.columns = prefix + self.test.columns + suffix
        return self
    
    @abstractmethod
    def create_features(self):
        raise NotImplementedError
    
    def save(self):
        # self.train.to_feather(str(self.train_path))
        # self.test.to_feather(str(self.test_path))
        logger.debug(f'save path={self.train_path}')
        logger.debug(f'save path={self.test_path}')
        logger.info(f'finish {self.name}')


def get_features(namespace):
    for k, v in namespace.items():
        if inspect.isclass(v) and issubclass(v, Feature) and not inspect.isabstract(v):
            yield v()




