"""第6章：LightGBM・XGBoost・CatBoost を、同じ入力（数値＋カテゴリ）と同じ程度の設定で比べる。"""
# %% Step 1 — データ｜開発期間の最後の折の全行（学習 2,232,108 行）。カテゴリはネイティブに渡す。
import time

import pandas as pd

from common import SEED, show, summary
from common.data import cv_folds
from common.features import load_top, split
from common.models.gbdt import fit_predict, frame, levels_of

df = load_top()
lv = levels_of(df)
f = cv_folds()[-1]
tr, va = split(df, *f["train"]), split(df, *f["valid"])
Xtr, Xva, ytr, yva = frame(tr, lv), frame(va, lv), tr.purchased.to_numpy(), va.purchased.to_numpy()

# %% Step 2 — 3 つのライブラリ｜木 300 本・学習率 0.05 にそろえ、木の大きさはそれぞれの既定の考え方で決める。
P = {"lgbm": dict(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=100),
     "xgb": dict(n_estimators=300, learning_rate=0.05, max_depth=6, min_child_weight=1),
     "cat": dict(iterations=300, learning_rate=0.05, depth=6)}
res = {}
for lib, params in P.items():
    t = time.time()
    p, _ = fit_predict(lib, params, Xtr, ytr, Xva, seed=SEED)
    print(f"[time] {lib} {time.time() - t:.1f} 秒")
    res[{"lgbm": "LightGBM", "xgb": "XGBoost", "cat": "CatBoost"}[lib]] = summary(yva, p)

# %% Step 3 — 比較｜第5章で採用した LR（AUC 0.8464・LogLoss 0.0433）を基準に、改善幅を見る。
show(pd.DataFrame(res).T, "購入の予測（検証 day 133〜146、学習 2,232,108 行、合成データ）")
