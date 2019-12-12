import pandas as pd
from logging import getLogger
from CONST import TRAIN_PATH, TEST_PATH

logger = getLogger('load_data')

def load_csv(path):
    logger.debug('enter')
    df = pd.read_csv(path)
    logger.debug('exit')
    return df

def load_train_data():
    logger.debug('enter')
    df = load_csv(TRAIN_PATH)
    logger.debug('exit')
    return df

def load_test_data():
    logger.debug('enter')
    df = load_csv(TEST_PATH)
    logger.debug('exit')
    return df


if __name__ == '__main__':
    print(load_train_data().head())
    print(load_test_data().head())
