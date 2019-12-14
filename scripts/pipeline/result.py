import os
import logging
import datetime
import sys
sys.path.append('../utils')
from util import Util
from CONST import RESUTL_DIR

logger = logging.getLogger('result')
logger.setLevel(logging.DEBUG)


class ResultHandler:

    def __init__(self, model_name):
        self.now = datetime.datetime.now()
        self.model_name = model_name
        self.name = ''
        self.result_dir = ''
        self.create_name()
        self.create_dir()

    def create_name(self):
        args = Util.get_arguments()
        prefix = 'debug_' if args.debug else ''

        now_form = "{0:%Y%m%d_%H%M%S}".format(self.now)
        self.name = f'{prefix}{now_form}_{self.model_name}'

    def create_dir(self):
        self.result_dir = f'{RESUTL_DIR}/{self.name}'
        os.makedirs(os.path.dirname(self.result_dir), exist_ok=True)


if __name__ == '__main__':
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    logger.addHandler(sh)

    # debug
    logger.debug('---- entry ----')
    rh = ResultHandler('xgb')
