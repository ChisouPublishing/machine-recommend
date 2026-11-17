"""第6章：勾配ブースティングの最小実装と、3 つのライブラリ（LightGBM・XGBoost・CatBoost）を同じ形で呼ぶ包み。"""
# %% Step 1 — 最小実装｜対数損失の勾配と 2 階微分で、回帰木を 1 本ずつ足していく（木の形は scikit-learn の回帰木に任せる）。
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from common.features import CAT_COLS, NUM_COLS


class MiniGBDT:
    def __init__(self, n_trees=100, lr=0.1, depth=4, min_leaf=200, l2=1.0, seed=0):
        self.n, self.lr, self.depth, self.min_leaf, self.l2, self.seed = n_trees, lr, depth, min_leaf, l2, seed

    def fit(self, X, y):
        self.f0 = np.log(y.mean() / (1 - y.mean()))
        f = np.full(len(y), self.f0)
        self.trees = []
        for t in range(self.n):
            p = 1 / (1 + np.exp(-f))
            g, h = p - y, p * (1 - p)                         # 勾配と 2 階微分（対数損失）
            tree = DecisionTreeRegressor(max_depth=self.depth, min_samples_leaf=self.min_leaf, random_state=self.seed + t)
            tree.fit(X, -g / np.maximum(h, 1e-6), sample_weight=h)   # ニュートン法の 1 歩を木で近似する
            leaf = tree.apply(X)
            val = {}                                          # 葉の値 = −Σg / (Σh + λ)
            for k in np.unique(leaf):
                m = leaf == k
                val[k] = -g[m].sum() / (h[m].sum() + self.l2)
            self.trees.append((tree, val))
            f += self.lr * np.vectorize(val.get)(leaf)
        return self

    def decision(self, X):
        f = np.full(X.shape[0], self.f0)
        for tree, val in self.trees:
            f += self.lr * np.vectorize(val.get)(tree.apply(X))
        return f

    def predict_proba(self, X):
        return 1 / (1 + np.exp(-self.decision(X)))


# %% Step 2 — 入力の形をそろえる｜数値はそのまま、カテゴリは全期間の値の一覧で固定した category 型にする（学習と検証で番号をそろえる）。
GB_CATS = [c for c in CAT_COLS if c not in ("position", "hour", "weekday")]
GB_COLS = NUM_COLS + ["position", "hour", "weekday"] + GB_CATS


def frame(df: pd.DataFrame, levels: dict) -> pd.DataFrame:
    X = df[GB_COLS].copy()
    for c in GB_CATS:
        X[c] = pd.Categorical(X[c].astype(str), categories=levels[c])
    return X


def levels_of(df: pd.DataFrame) -> dict:
    return {c: sorted(df[c].astype(str).unique()) for c in GB_CATS}


# %% Step 3 — 3 つのライブラリ｜同じ入力・同じ乱数の種で学習し、予測確率を返す。params は Optuna の提案をそのまま受ける。
def fit_predict(lib: str, params: dict, Xtr, ytr, Xva, seed: int = 0, threads: int = 8):
    if lib == "lgbm":
        import lightgbm as lgb
        m = lgb.LGBMClassifier(objective="binary", random_state=seed, n_jobs=threads, verbose=-1, deterministic=True,
                               force_row_wise=True, **params)          # 並列でも結果を毎回同じにする
        m.fit(Xtr, ytr)
        return m.predict_proba(Xva)[:, 1], m
    if lib == "xgb":
        import xgboost as xgb
        m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, max_cat_to_onehot=1, random_state=seed,
                              n_jobs=threads, eval_metric="logloss", **params)
        m.fit(Xtr, ytr)
        return m.predict_proba(Xva)[:, 1], m
    if lib == "cat":
        from catboost import CatBoostClassifier, Pool
        cats = [Xtr.columns.get_loc(c) for c in GB_CATS]
        to_str = lambda X: X.astype({c: str for c in GB_CATS})
        m = CatBoostClassifier(loss_function="Logloss", random_seed=seed, thread_count=threads, verbose=False,
                               allow_writing_files=False, **params)
        m.fit(Pool(to_str(Xtr), ytr, cat_features=cats))
        return m.predict_proba(Pool(to_str(Xva), cat_features=cats))[:, 1], m
    raise ValueError(lib)


# %% Step 4 — 負例の間引きと補正｜負例を率 r で残して学習した確率 q を、元の分布の確率 p = q / (q + (1 − q)/r) に戻す。
def downsample(df: pd.DataFrame, label: str, r: float, seed: int = 0) -> pd.DataFrame:
    keep = (df[label] == 1) | (np.random.default_rng(seed).random(len(df)) < r)
    return df[keep]


def correct(q: np.ndarray, r: float) -> np.ndarray:
    return q / (q + (1 - q) / r)
