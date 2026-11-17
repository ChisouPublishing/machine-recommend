"""第3章：ID のカテゴリ変数（商品・ブランド）の 4 つの扱い方を、同じロジスティック回帰と LightGBM で比べる。"""
# %% Step 1 — 準備｜開発期間の最後の折で、学習 30 万行・検証の全行を使う（比べやすい大きさに抜き出す）。
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import SEED, auc, logloss, show
from common.data import cv_folds
from common.features import NUM_COLS, load_top, split

df = load_top()
f = cv_folds()[-1]
tr = split(df, *f["train"]).sample(300_000, random_state=SEED)
va = split(df, *f["valid"])
ID = ["item_id", "brand_id", "category_id"]
y_tr, y_va = tr.clicked.values, va.clicked.values
sc = StandardScaler().fit(np.log1p(tr[NUM_COLS].clip(lower=0)))
num_tr = sparse.csr_matrix(sc.transform(np.log1p(tr[NUM_COLS].clip(lower=0))))
num_va = sparse.csr_matrix(sc.transform(np.log1p(va[NUM_COLS].clip(lower=0))))
rows = []

def lr(Xtr, Xva, name):
    t = time.time()
    m = LogisticRegression(C=1.0, max_iter=300).fit(Xtr, y_tr)
    p = m.predict_proba(Xva)[:, 1]
    rows.append((name, Xtr.shape[1], auc(y_va, p), logloss(y_va, p), time.time() - t))

# %% Step 2 — ワンホット｜ID ごとに 1 列。学習に無い ID は全部 0 になる。
oh = OneHotEncoder(handle_unknown="ignore", min_frequency=5).fit(tr[ID].astype(str))
lr(sparse.hstack([num_tr, oh.transform(tr[ID].astype(str))]).tocsr(),
   sparse.hstack([num_va, oh.transform(va[ID].astype(str))]).tocsr(), "LR＋ワンホット")

# %% Step 3 — ハッシュ｜「列名=値」の文字列を 2^14 列に写す。列数を固定でき、新しい ID もどこかの列に入る。
fh = FeatureHasher(n_features=2 ** 14, input_type="string")
hs = lambda d: fh.transform([[f"{c}={v}" for c, v in zip(ID, r)] for r in d[ID].itertuples(index=False)])
lr(sparse.hstack([num_tr, hs(tr)]).tocsr(), sparse.hstack([num_va, hs(va)]).tocsr(), "LR＋ハッシュ（2^14）")

# %% Step 4 — ターゲットエンコーディング｜ID ごとの過去のクリック率。学習期間の前半で作った率を後半の学習に使い、リークを防ぐ。
cut = tr.day.quantile(0.5)
early, late = tr[tr.day <= cut], tr[tr.day > cut]
prior, k = early.clicked.mean(), 50

def te(src, dst):
    out = []
    for c in ID:
        g = src.groupby(c).clicked.agg(["sum", "count"])
        rate = (g["sum"] + prior * k) / (g["count"] + k)          # 件数が少ない ID は全体の率に寄せる（平滑化）
        out.append(dst[c].map(rate).fillna(prior).to_numpy())
    return np.column_stack(out)

sc2 = StandardScaler().fit(np.log1p(late[NUM_COLS].clip(lower=0)))
X_late = np.hstack([sc2.transform(np.log1p(late[NUM_COLS].clip(lower=0))), te(early, late)])
X_va = np.hstack([sc2.transform(np.log1p(va[NUM_COLS].clip(lower=0))), te(tr, va)])
y_tr_save = y_tr
y_tr = late.clicked.values
lr(X_late, X_va, "LR＋ターゲットエンコーディング")
y_tr = y_tr_save

# %% Step 5 — ネイティブ｜LightGBM にカテゴリの列として渡す。木の分割でカテゴリをまとめて扱う。
t = time.time()
X = lambda d: d[NUM_COLS + ID].astype({c: "category" for c in ID})
cats = {c: sorted(df[c].unique()) for c in ID}
Xc = lambda d: X(d).astype({c: pd.CategoricalDtype(cats[c]) for c in ID})
g = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31, random_state=SEED, verbose=-1)
g.fit(Xc(tr), y_tr)
p = g.predict_proba(Xc(va))[:, 1]
rows.append(("LightGBM＋ネイティブ", len(NUM_COLS) + len(ID), auc(y_va, p), logloss(y_va, p), time.time() - t))

# %% Step 6 — 比較｜列数・検証の AUC と LogLoss を並べる（学習時間は環境で変わるので本文に載せない）。
res = pd.DataFrame(rows, columns=["扱い方", "列数", "AUC", "LogLoss", "秒"])
for r in res.itertuples():
    print(f"[time] {r.扱い方} {r.秒:.1f} 秒")
show(res.drop(columns="秒"), "ID の扱い方の比較（クリック、検証 day 133〜146、合成データ）", index=False)
