import numpy as np
import pandas as pd
import logging
from model import Model
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from typing import Callable, List, Optional, Tuple, Union

# logger
logger = logging.getLogger('runner')
logger.setLevel(logging.DEBUG)


class Runner:

    def __init__(self, run_name: str, model_cls: Callable[[str, dict], Model], features: List[str], params: dict):
        """コンストラクタ
        :param run_name: ランの名前
        :param model_cls: モデルのクラス
        :param features: 特徴量のリスト
        :param params: ハイパーパラメータ
        """
        self.run_name = run_name
        self.model_cls = model_cls
        self.features = features
        self.params = params
        self.n_fold = 4

    def train_fold(self, i_fold: int) -> Tuple[
        Model, np.array, np.array, float]:
        """クロスバリデーションでのfoldを指定して学習・評価を行う
        他のメソッドから呼び出すほか、単体でも確認やパラメータ調整に用いる
        :param i_fold: foldの番号
        :return: （モデルのインスタンス、レコードのインデックス、予測値、評価によるスコア）のタプル
        """
        # 学習データの読込
        train_x = self.load_x_train()
        train_y = self.load_y_train()

        # 学習データ・バリデーションデータをセットする
        tr_idx, va_idx = self.load_index_fold(i_fold)
        tr_x, tr_y = train_x.iloc[tr_idx], train_y.iloc[tr_idx]
        va_x, va_y = train_x.iloc[va_idx], train_y.iloc[va_idx]

        # 学習を行う
        model = self.build_model(i_fold)
        model.train(tr_x, tr_y, va_x, va_y)

        # バリデーションデータへの予測・評価を行う
        va_pred = model.predict(va_x)
        score = log_loss(va_y, va_pred, eps=1e-15, normalize=True)

        # モデル、インデックス、予測値、評価を返す
        return model, va_idx, va_pred, score


def run_train_cv(self) -> None:
        """クロスバリデーションでの学習・評価を行う
        学習・評価とともに、各foldのモデルの保存、スコアのログ出力についても行う
        """
        logger.info(f'{self.run_name} - start training cv')

        scores = []
        va_idxes = []
        preds = []

        # 各foldで学習を行う
        for i_fold in range(self.n_fold):
            # 学習を行う
            logger.info(f'{self.run_name} fold {i_fold} - start training')
            model, va_idx, va_pred, score = self.train_fold(i_fold)
            logger.info(f'{self.run_name} fold {i_fold} - end training - score {score}')

            # モデルを保存する
            # model.save_model()

            # 結果を保持する
            va_idxes.append(va_idx)
            scores.append(score)
            preds.append(va_pred)

        # 各foldの結果をまとめる
        va_idxes = np.concatenate(va_idxes)
        order = np.argsort(va_idxes)
        preds = np.concatenate(preds, axis=0)
        preds = preds[order]

        logger.info(f'{self.run_name} - end training cv - score {np.mean(scores)}')

        # 予測結果の保存
        # Util.dump(preds, f'../model/pred/{self.run_name}-train.pkl')

        # 評価結果の保存
        # logger.result_scores(self.run_name, scores)

