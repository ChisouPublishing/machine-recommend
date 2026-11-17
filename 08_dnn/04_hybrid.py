"""第8章：GBDT と DNN を組み合わせる 2 つの方法（GBDT の葉を特徴量にする・予測を混ぜる）を、単独のモデルと比べる。"""
# %% Step 1 — データ｜学習 day 28〜125、混ぜる重みを決める期間 day 126〜132、評価 day 133〜146（負例 10%、予測は補正）。
import common  # noqa: F401  先に読み込む（LightGBM を PyTorch より先に。付録 A）
import json

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from common import OUT, SEED, logloss, show, summary
from common.features import NUM_COLS, load_top, split
from common.models.calibration import logit
from common.models.gbdt import correct, downsample, fit_predict, frame, levels_of
from common.models.torch_ctr import DeepFM, Encoder, train

torch.set_num_threads(1)   # 並列の足し合わせの順で結果が揺れないよう、1 スレッドで学習する
R = 0.1
df = load_top()
lv = levels_of(df)
tr = downsample(split(df, 28, 125), "purchased", R, seed=SEED)
ho, va = split(df, 126, 132), split(df, 133, 146)
both = pd.concat([ho, va])
y_ho, y_va = ho.purchased.to_numpy(), va.purchased.to_numpy()
lg = json.loads((OUT / "06_gbdt" / "best_gbdt.json").read_text())["lgbm"]
dn = json.loads((OUT / "08_dnn" / "best_dnn.json").read_text())["mine:DeepFM"]

# %% Step 2 — 単独のモデル｜第6章の LightGBM（最良の構成）と、第8章の DeepFM（最小実装、最良の構成）。
q, gb = fit_predict("lgbm", lg, frame(tr, lv), tr.purchased.to_numpy(), frame(both, lv), seed=SEED)
p_gb = correct(q, R)
enc = Encoder().fit(tr)
m = DeepFM(enc.sizes(), len(NUM_COLS), k=dn["k"], hidden=(dn["h1"], dn["h1"] // 2), dropout=dn["dropout"])
p_dn = correct(train(m, enc.transform(tr), tr.purchased.to_numpy(), enc.transform(both), epochs=dn["epochs"], lr=dn["lr"],
                     weight_decay=dn["wd"], seed=SEED), R)
n = len(ho)

# %% Step 3 — GBDT＋LR｜LightGBM の各木で落ちた葉の番号をカテゴリとしてワンホットにし、ロジスティック回帰に入れる（Facebook の 2014 年の方法）。
leaf = lambda d: gb.booster_.predict(frame(d, lv), pred_leaf=True)
oh = OneHotEncoder(handle_unknown="ignore").fit(leaf(tr))
lr = LogisticRegression(C=0.05, max_iter=500).fit(oh.transform(leaf(tr)), tr.purchased)
p_gl = correct(lr.predict_proba(oh.transform(leaf(both)))[:, 1], R)

# %% Step 4 — 予測を混ぜる｜対数オッズの重み付きの和 w·logit(GBDT) + (1−w)·logit(DNN)。重み w は day 126〜132 の LogLoss で決める。
ws = np.linspace(0, 1, 21)
mix = lambda w, a, b: 1 / (1 + np.exp(-(w * logit(a) + (1 - w) * logit(b))))
w = ws[np.argmin([logloss(y_ho, mix(w, p_gb[:n], p_dn[:n])) for w in ws])]
print(f"混ぜる重み（GBDT の側）: {w:.2f}")

# %% Step 5 — 比較｜評価の期間で、4 つを並べる。
res = {"LightGBM（単独）": p_gb[n:], "DeepFM（単独）": p_dn[n:], "GBDT の葉＋LR": p_gl[n:], f"混ぜる（w={w:.2f}）": mix(w, p_gb[n:], p_dn[n:])}
show(pd.DataFrame({k: summary(y_va, p) for k, p in res.items()}).T, "GBDT と DNN の組み合わせ（評価 day 133〜146、合成データ）")
