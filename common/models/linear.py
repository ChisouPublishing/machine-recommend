"""第5章：線形モデルの最小実装（numpy のロジスティック回帰、FTRL-Proximal）と、特徴量のハッシュ化。"""
# %% Step 1 — 数値の特徴量｜件数の偏った列を log(1+x) で縮め、学習期間の平均と標準偏差で標準化する。
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from common.features import NUM_COLS


class NumScaler:
    def fit(self, df: pd.DataFrame):
        x = np.log1p(df[NUM_COLS].clip(lower=0).to_numpy(float))
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-9
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return (np.log1p(df[NUM_COLS].clip(lower=0).to_numpy(float)) - self.mu) / self.sd


# %% Step 2 — ハッシュ化｜カテゴリと交差（例：カテゴリ×性別）を「名前=値」の文字列にしてハッシュし、2^bits 列の疎な行列にする。
HASH_COLS = ["item_id", "brand_id", "category_id", "major", "position", "hour", "weekday", "member_rank", "device",
             "body_size", "season", "price_tier", "item_gender", "age_band", "gender"]
CROSSES = [("category_id", "gender"), ("category_id", "age_band"), ("brand_id", "gender"), ("major", "body_size"),
           ("season", "weekday")]


def hash_codes(df: pd.DataFrame) -> np.ndarray:
    """各行・各特徴の 64 ビットのハッシュ（行数, 特徴数）。bits を変えるときは剰余を取り直すだけで済む。"""
    out = []
    for c in HASH_COLS:
        out.append(pd.util.hash_array((c + "=" + df[c].astype(str)).to_numpy(dtype=object)))
    for a, b in CROSSES:
        s = a + "×" + b + "=" + df[a].astype(str) + "|" + df[b].astype(str)
        out.append(pd.util.hash_array(s.to_numpy(dtype=object)))
    out.append(pd.util.hash_array(np.full(len(df), "bias", dtype=object)))     # 切片（すべての行で 1）
    return np.column_stack(out)


def design(df: pd.DataFrame, scaler: NumScaler, codes: np.ndarray, bits: int) -> sparse.csr_matrix:
    """数値（標準化）とハッシュしたカテゴリを横に並べた、線形モデルの入力。"""
    return sparse.hstack([sparse.csr_matrix(scaler.transform(df)), hashed_matrix(codes, bits)]).tocsr()


def hashed_matrix(codes: np.ndarray, bits: int) -> sparse.csr_matrix:
    n, k = codes.shape
    idx = (codes % np.uint64(2 ** bits)).astype(np.int64).ravel()
    return sparse.csr_matrix((np.ones(n * k), idx, np.arange(0, n * k + 1, k)), shape=(n, 2 ** bits))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


# %% Step 3 — numpy のロジスティック回帰｜損失 = 対数損失の平均 + (λ/2)|w|²。全件の勾配で、Adam で更新する。
class NumpyLR:
    def __init__(self, l2: float = 1e-4, lr: float = 0.05, epochs: int = 300):
        self.l2, self.lr, self.epochs = l2, lr, epochs

    def fit(self, X: np.ndarray, y: np.ndarray):
        n, d = X.shape
        self.w, self.b = np.zeros(d), np.log(y.mean() / (1 - y.mean()))
        m, v = np.zeros(d + 1), np.zeros(d + 1)
        for t in range(1, self.epochs + 1):
            p = sigmoid(X @ self.w + self.b)
            g = np.r_[X.T @ (p - y) / n + self.l2 * self.w, (p - y).mean()]
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            step = self.lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
            self.w -= step[:-1]
            self.b -= step[-1]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(X @ self.w + self.b)


# %% Step 4 — FTRL-Proximal｜座標ごとに学習率を変え、L1 で多くの重みをちょうど 0 にする（広告の CTR 予測で広く使われた方法）。
class FTRL:
    """McMahan et al. (2013) の FTRL-Proximal。本書は 1 行ずつではなく、小さなまとまり（batch 行）ごとに更新する。"""

    def __init__(self, alpha=0.05, beta=1.0, l1=1.0, l2=1.0, batch=1024, epochs=1, seed=0):
        self.a, self.bt, self.l1, self.l2, self.batch, self.epochs, self.seed = alpha, beta, l1, l2, batch, epochs, seed

    def _w(self):
        w = -(self.z - np.sign(self.z) * self.l1) / ((self.bt + np.sqrt(self.n)) / self.a + self.l2)
        return np.where(np.abs(self.z) <= self.l1, 0.0, w)

    def fit(self, X: sparse.csr_matrix, y: np.ndarray):
        d = X.shape[1]
        self.z, self.n = np.zeros(d), np.zeros(d)
        r = np.random.default_rng(self.seed)
        for _ in range(self.epochs):
            for s in range(0, X.shape[0], self.batch):
                xb, yb = X[s:s + self.batch], y[s:s + self.batch]
                w = self._w()
                g = xb.T @ (sigmoid(xb @ w) - yb)                 # まとまりの勾配（座標ごと）
                sigma = (np.sqrt(self.n + g * g) - np.sqrt(self.n)) / self.a
                self.z += g - sigma * w
                self.n += g * g
        self.w = self._w()
        return self

    def predict_proba(self, X: sparse.csr_matrix) -> np.ndarray:
        return sigmoid(X @ self.w)
