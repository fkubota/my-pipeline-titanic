import sys
import xgboost as xgb
from model import Model
sys.path.append('../utils')
from util import Util


class ModelXGB(Model):

    def train(self, tr_x, tr_y, va_x=None, va_y=None):

        # データのセット
        validation = va_x is not None
        dtrain = xgb.DMatrix(tr_x, label=tr_y)
        if validation:
            dvalid = xgb.DMatrix(va_x, label=va_y)

        # ハイパーパラメータの設定
        params = dict(self.params)
        num_round = params.pop('num_round')

        # 学習
        if validation:
            early_stopping_rounds = params.pop('early_stopping_rounds')
            watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
            self.model = xgb.train(params, dtrain, num_round, evals=watchlist,
                                   early_stopping_rounds=early_stopping_rounds)
        else:
            watchlist = [(dtrain, 'train')]
            self.model = xgb.train(params, dtrain, num_round, evals=watchlist)

    def predict(self, te_x):
        dtest = xgb.DMatrix(te_x)
        return self.model.predict(dtest,
                                  ntree_limit=self.model.best_ntree_limit)

    def save_model(self, save_dir):
        save_path = f'{save_dir}/{self.run_fold_name}.pkl'
        Util.save(self.model, save_path)

    def load_model(self, save_dir):
        load_path = f'{save_dir}/{self.run_fold_name}.pkl'
        self.model = Util.load(load_path)
