import sys
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from model import Model
from result import ResultHandler
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from typing import Callable, List, Tuple, Union
from analysys import permutation_importance
sys.path.append('../utils')
from util import Util
mpl.rcParams['axes.facecolor'] = 'ffffff'


# logger
logger = logging.getLogger('runner')
logger.setLevel(logging.DEBUG)
load_data_logger = logging.getLogger('util')
load_data_logger.setLevel(logging.DEBUG)


class Runner:

    def __init__(self, model_cls: Callable[[str, dict], Model],
                 n_fold: int,
                 feat_grps: List[str],
                 params: dict,
                 result_handler: ResultHandler):
        """コンストラクタ
        :param run_name: ランの名前
        :param model_cls: モデルのクラス
        :param feat_grps: 特徴量グループ名
        :param params: ハイパーパラメータ
        """
        self.model_cls = model_cls
        self.feat_grps = feat_grps
        self.params = params
        self.n_fold = n_fold
        self.result_handler = result_handler
        self.run_name = self.result_handler.name
        self.pi_results = []

        # save_params
        self.save_run_params()

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
        # score = log_loss(va_y, va_pred, eps=1e-15, normalize=True)
        score = f1_score(va_y, va_pred)

        # permutation importance
        pi = permutation_importance(model, f1_score)
        pi.compute(va_x, va_y)
        self.pi_results.append(pi.df_result)

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
            logger.info(f'{self.run_name} fold {i_fold}'
                        f' - end training - valid-score {score:.4f}')

            # モデルを保存する
            model.save_model(self.result_handler.result_dir)
            logger.debug(f'save model {i_fold}')

            # 結果を保持する
            va_idxes.append(va_idx)
            scores.append(score)
            preds.append(va_pred)

        # 各foldの結果をまとめる
        va_idxes = np.concatenate(va_idxes)
        order = np.argsort(va_idxes)
        preds = np.concatenate(preds, axis=0)
        preds = preds[order]

        # permmutation_importance をまとめる
        df_concat = pd.concat(self.pi_results)
        n_feat = len(df_concat.feat.unique())
        plt.figure(figsize=(10, int(1*n_feat)), dpi=100)
        sns.barplot(x="score_diff", y='feat',
                    data=df_concat.sort_values(by='score_diff',
                                               ascending=True))
        result_dir = self.result_handler.result_dir
        plt.savefig(f'{result_dir}/permutation_importance.png')

        logger.info(f'{self.run_name} - end training cv'
                    f' - oof score mean:{np.mean(scores):.4f}, '
                    f'std:{np.std(scores):.4f}')

        # oofの保存
        Util.save_oof(preds, self.result_handler.result_dir)

        # 評価結果の保存
        # logger.result_scores(self.run_name, scores)

    def run_predict_cv(self) -> None:
        """クロスバリデーションで学習した各foldのモデルの平均により、テストデータの予測を行う
        あらかじめrun_train_cvを実行しておく必要がある
        """
        logger.info(f'{self.run_name} - start prediction cv')

        test_x = self.load_x_test()

        preds = []

        # 各foldのモデルで予測を行う
        for i_fold in range(self.n_fold):
            # logger.info(f'{self.run_name} - start prediction fold:{i_fold}')
            model = self.build_model(i_fold)
            model.load_model(self.result_handler.result_dir)
            pred = model.predict(test_x)
            preds.append(pred)
            logger.info(f'{self.run_name}'
                        f' - end prediction fold:{i_fold}/{self.n_fold-1}')

        # 予測の平均値を出力する
        pred_avg = np.mean(preds, axis=0)

        # 予測結果の保存
        Util.save_submission(pred_avg, self.result_handler.result_dir)

        logger.info(f'{self.run_name} - end prediction cv')

    def build_model(self, i_fold: Union[int, str]) -> Model:
        """クロスバリデーションでのfoldを指定して、モデルの作成を行う
        :param i_fold: foldの番号
        :return: モデルのインスタンス
        """
        # ラン名、fold、モデルのクラスからモデルを作成する
        run_fold_name = f'{self.run_name}_model-{i_fold}'
        return self.model_cls(run_fold_name, self.params)

    def load_x_train(self) -> pd.DataFrame:
        """学習データの特徴量を読み込む
        :return: 学習データの特徴量
        """
        # 学習データの読込を行う
        # 列名で抽出する以上のことを行う場合、このメソッドの修正が必要
        # 毎回train.csvを読み込むのは効率が悪いため、データに応じて適宜対応するのが望ましい（他メソッドも同様）
        return Util.load_train_features(self.feat_grps)

    def load_y_train(self) -> pd.Series:
        """学習データの目的変数を読み込む
        :return: 学習データの目的変数
        """
        # 目的変数の読込を行う
        train_y = Util.load_train_data()['Survived']
        train_y = pd.Series(train_y)
        return train_y

    def load_x_test(self) -> pd.DataFrame:
        """テストデータの特徴量を読み込む
        :return: テストデータの特徴量
        """
        return Util.load_test_features(self.feat_grps)

    def load_index_fold(self, i_fold: int) -> np.array:
        """クロスバリデーションでのfoldを指定して対応するレコードのインデックスを返す
        :param i_fold: foldの番号
        :return: foldに対応するレコードのインデックス
        """
        # 学習データ・バリデーションデータを分けるインデックスを返す
        # ここでは乱数を固定して毎回作成しているが、ファイルに保存する方法もある
        train_y = self.load_y_train()
        dummy_x = np.zeros(len(train_y))
        skf = StratifiedKFold(n_splits=self.n_fold,
                              shuffle=True, random_state=71)
        return list(skf.split(dummy_x, train_y))[i_fold]

    def save_run_params(self):
        params = {
                  'n_fold': self.n_fold,
                  'feat_grps': self.feat_grps,
                  'model_params': self.params
                  }
        Util.save_params(params, self.result_handler.result_dir)
