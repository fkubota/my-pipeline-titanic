import logging
import datetime

logger = logging.getLogger('result')
logger.setLevel(logging.DEBUG)


class ResultHandler:

    def __init__(self, model_name):
        self.now = datetime.datetime.now()
        self.model_name = model_name
        self.name = ''
        self.create_name()

    def create_name(self):
        now_form = "{0:%Y%m%d_%H%M%S}".format(self.now)
        self.name = f'{now_form}_{self.model_name}'


if __name__ == '__main__':
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    logger.addHandler(sh)

    # debug
    logger.debug('---- entry ----')
    rh = ResultHandler('xgb')
    print(rh.name)
