import sys
import lightgbm as lgb
from model import Model
sys.path.append('../utils')
from util import Util


class ModelLGBM(Model):

    def train(self, tr_x, tr_y, va_x=None, va_y=None):
        # set params
        params = dict(self.params)

        # 学習
        self.model = lgb.LGBMClassifier(**params)
        self.model.fit(tr_x, tr_y,
                       eval_set=[(tr_x, tr_y), (va_x, va_y)],
                       early_stopping_rounds=100,
                       verbose=20)

    def predict(self, te_x):
        return self.model.predict(te_x,
                                  num_iteration=self.model.best_iteration_)

    def save_model(self, save_dir):
        save_path = f'{save_dir}/{self.run_fold_name}.pkl'
        Util.save(self.model, save_path)

    def load_model(self, save_dir):
        load_path = f'{save_dir}/{self.run_fold_name}.pkl'
        self.model = Util.load(load_path)
