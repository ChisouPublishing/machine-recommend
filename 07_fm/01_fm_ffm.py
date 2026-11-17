"""第7章：FM・FFM を、最小実装（PyTorch）・FuxiCTR・myFM（ベイズ FM）で比べる（学習は負例を 10% に間引き、予測は補正）。"""
# %% Step 1 — データ｜開発期間の最後の折。学習 day 28〜125、早期終了用 day 126〜132（FuxiCTR）、評価 day 133〜146。
import common  # noqa: F401  先に読み込む（LightGBM を PyTorch より先に。付録 A）
import time

import numpy as np
import pandas as pd
import torch

from common import SEED, show, summary
from common.features import NUM_COLS, load_top, split
from common.models import fuxi
from common.models.gbdt import correct, downsample
from common.models.torch_ctr import FFM, FM, FIELDS, Encoder, train

torch.set_num_threads(1)   # 並列の足し合わせの順で結果が揺れないよう、1 スレッドで学習する
R = 0.1
df = load_top()
tr = downsample(split(df, 28, 132), "purchased", R, seed=SEED)
va = split(df, 133, 146)
y = va.purchased.to_numpy()
enc = Encoder().fit(tr)
xtr, xva = enc.transform(tr), enc.transform(va)
res, secs = {}, {}

# %% Step 2 — 最小実装｜FM（埋め込み 8 次元）と FFM（相手の列ごとに 4 次元）を、Adam で 3 エポック学習する。
for name, M, kw in (("FM（最小実装）", FM, {"k": 8}), ("FFM（最小実装）", FFM, {"k": 4})):
    t = time.time()
    m = M(enc.sizes(), len(NUM_COLS), **kw)
    res[name] = summary(y, correct(train(m, xtr, tr.purchased.to_numpy(), xva, epochs=3, lr=5e-3, seed=SEED), R))
    secs[name] = time.time() - t

# %% Step 3 — FuxiCTR｜同じ行で FM・FFM を学習する。数値は区間に分け、系列は渡さない（02 で確かめる理由による）。
#               学習の末の 7 日を早期終了の判断に使う（評価の期間は使わない）。
ftr, fho = downsample(split(df, 28, 125), "purchased", R, seed=SEED), downsample(split(df, 126, 132), "purchased", R, seed=SEED)
data = fuxi.prepare("ch07", ftr, fho, va)
for name in ("FM", "FFM"):
    t = time.time()
    p = fuxi.fit_predict(name, data, {"embedding_dim": 8 if name == "FM" else 4, "learning_rate": 5e-3, "epochs": 3}, seed=SEED)
    res[f"{name}（FuxiCTR）"] = summary(y, correct(p, R))
    secs[f"{name}（FuxiCTR）"] = time.time() - t

# %% Step 4 — myFM｜ベイズ推定（ギブスサンプリング）の FM。カテゴリはワンホット、数値はそのまま疎な行列にする。
import myfm
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder

oh = OneHotEncoder(handle_unknown="ignore").fit(tr[FIELDS].astype(str))
X = lambda d: sparse.hstack([oh.transform(d[FIELDS].astype(str)), sparse.csr_matrix(enc.scaler.transform(d))]).tocsr()
t = time.time()
bfm = myfm.MyFMClassifier(rank=8, random_seed=SEED)
bfm.fit(X(tr), tr.purchased.to_numpy(), n_iter=60, n_kept_samples=40)
res["FM（myFM、ベイズ）"] = summary(y, correct(bfm.predict_proba(X(va)), R))
secs["FM（myFM、ベイズ）"] = time.time() - t

# %% Step 5 — 比較｜第5章の LR（AUC 0.8464）・第6章の LightGBM と比べる。学習の秒数は環境で変わるので本文に載せない。
for k, v in secs.items():
    print(f"[time] {k} {v:.1f} 秒")
show(pd.DataFrame(res).T, "FM・FFM の比較（評価 day 133〜146、学習は負例 10%、合成データ）")
