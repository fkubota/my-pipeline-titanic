import sys
import datetime
import logging

logger = logging.getLogger('result')
logger.setLevel(logging.DEBUG)


class ResultHandler:

    def __inint__(self):
        self.me = 'fkubota'
        self.datetime = datetime.datetme.now()

    def show_date(self):
        print(self.datetime)


if __name__ == '__main__':
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    logger.addHandler(sh)

    # debug
    logger.debug('---- entry ----')
    rh = ResultHandler
    print(rh.me)
