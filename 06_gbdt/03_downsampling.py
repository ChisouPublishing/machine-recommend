"""第6章：負例の間引き（negative downsampling）で学習を速くし、確率の補正で元の水準に戻す。"""
# %% Step 1 — データ｜開発期間の最後の折。購入は約 1% なので、負例を 10% だけ残すと行数は約 1/9 になる。
import time

import pandas as pd

from common import SEED, show, summary
from common.data import cv_folds
from common.features import load_top, split
from common.models.gbdt import correct, downsample, fit_predict, frame, levels_of

df = load_top()
lv = levels_of(df)
f = cv_folds()[-1]
tr, va = split(df, *f["train"]), split(df, *f["valid"])
yva = va.purchased.to_numpy()
P = dict(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=100)
rows = {}

# %% Step 2 — 間引きの率を変えて学習｜r = 1（全件）・0.3・0.1・0.03。補正する前と後の指標を並べる。
for r in (1.0, 0.3, 0.1, 0.03):
    d = downsample(tr, "purchased", r, seed=SEED)
    t = time.time()
    q, _ = fit_predict("lgbm", P, frame(d, lv), d.purchased.to_numpy(), frame(va, lv), seed=SEED)
    print(f"[time] r={r} {time.time() - t:.1f} 秒")
    rows[(r, len(d), "補正前")] = summary(yva, q)
    if r < 1:
        rows[(r, len(d), "補正後")] = summary(yva, correct(q, r))

# %% Step 3 — 比較｜AUC は補正で変わらない（順位は同じ）。平均の予測・LogLoss・ECE が補正で元に戻るかを見る。
out = pd.DataFrame(rows).T
out.index.names = ["残す率 r", "学習の行数", ""]
show(out, "負例の間引きと補正（LightGBM、検証 day 133〜146、合成データ）")
