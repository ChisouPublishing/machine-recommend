"""第6章：勾配ブースティングの最小実装を、同じ設定の LightGBM と比べる（数値の特徴量 20 列、学習は 30 万行）。"""
# %% Step 1 — データ｜開発期間の最後の折。最小実装は遅いので、学習は 30 万行に抜き出す（LightGBM も同じ行で学習する）。
import time

import lightgbm as lgb
import pandas as pd

from common import SEED, show, summary
from common.data import cv_folds
from common.features import NUM_COLS, load_top, split
from common.models.gbdt import MiniGBDT

df = load_top()
f = cv_folds()[-1]
tr = split(df, *f["train"]).sample(300_000, random_state=SEED)
va = split(df, *f["valid"])
Xtr, Xva, ytr, yva = tr[NUM_COLS].to_numpy(float), va[NUM_COLS].to_numpy(float), tr.purchased.to_numpy(), va.purchased.to_numpy()

# %% Step 2 — 最小実装｜深さ 4 の木を 100 本、学習率 0.1 で足す。
t = time.time()
mini = MiniGBDT(n_trees=100, lr=0.1, depth=4, min_leaf=200, l2=1.0, seed=SEED).fit(Xtr, ytr)
print(f"[time] 最小実装 {time.time() - t:.1f} 秒")

# %% Step 3 — LightGBM｜同じ木の数・学習率・深さ・葉の最小の行数・L2 で学習する（木の作り方はヒストグラム法）。
t = time.time()
lg = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, num_leaves=16, min_child_samples=200,
                        reg_lambda=1.0, random_state=SEED, verbose=-1).fit(Xtr, ytr)
print(f"[time] LightGBM {time.time() - t:.1f} 秒")

# %% Step 4 — 比較｜最小実装と LightGBM の検証の指標。第5章の数値だけの LR（AUC 0.8432）とも比べる。
show(pd.DataFrame({"最小実装（100 本）": summary(yva, mini.predict_proba(Xva)),
                   "LightGBM（100 本）": summary(yva, lg.predict_proba(Xva)[:, 1])}).T,
     "購入の予測（検証 day 133〜146、数値 20 列、学習 30 万行、合成データ）")
