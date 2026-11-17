"""第5章：カテゴリと交差をハッシュした疎な特徴量で、FTRL（最小実装）とライブラリ（SGD・LBFGS）を比べる。"""
# %% Step 1 — データ｜数値 20 列とハッシュ（2^20 列）を横に並べる。1 行あたりの非ゼロは 41 個。
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier

from common import SEED, show, summary
from common.data import cv_folds
from common.features import load_top, split
from common.models.linear import FTRL, NumScaler, design, hash_codes

df = load_top()
f = cv_folds()[-1]
tr, va = split(df, *f["train"]), split(df, *f["valid"])
sc = NumScaler().fit(tr)
BITS = 20
Xtr, Xva = design(tr, sc, hash_codes(tr), BITS), design(va, sc, hash_codes(va), BITS)
ytr, yva = tr.purchased.to_numpy(), va.purchased.to_numpy()
print("学習の行列:", Xtr.shape, "非ゼロ/行:", Xtr.nnz // Xtr.shape[0])
res, nz = {}, {}

# %% Step 2 — 最小実装：FTRL｜1,024 行ずつ、座標ごとの学習率で更新する。L1 で、出現の少ない列の重みは 0 のままになる。
t = time.time()
ftrl = FTRL(alpha=0.05, beta=1.0, l1=1.0, l2=1.0, batch=1024, epochs=2).fit(Xtr, ytr)
print(f"[time] FTRL {time.time() - t:.1f} 秒")
res["FTRL（最小実装）"], nz["FTRL（最小実装）"] = summary(yva, ftrl.predict_proba(Xva)), int((ftrl.w != 0).sum())

# %% Step 3 — ライブラリ：SGD｜scikit-learn の SGDClassifier（対数損失、Elastic Net）。大規模な疎データの定番。
t = time.time()
sgd = SGDClassifier(loss="log_loss", penalty="elasticnet", alpha=1e-6, l1_ratio=0.15, max_iter=5, tol=None,
                    random_state=SEED).fit(Xtr, ytr)
print(f"[time] SGD {time.time() - t:.1f} 秒")
res["SGD（scikit-learn）"], nz["SGD（scikit-learn）"] = summary(yva, sgd.predict_proba(Xva)[:, 1]), int((sgd.coef_ != 0).sum())

# %% Step 4 — ライブラリ：LBFGS｜L2 正則化のロジスティック回帰を、準ニュートン法で収束まで解く（基準）。
t = time.time()
lb = LogisticRegression(C=0.1, max_iter=300).fit(Xtr, ytr)
print(f"[time] LBFGS {time.time() - t:.1f} 秒")
res["LBFGS（scikit-learn）"], nz["LBFGS（scikit-learn）"] = summary(yva, lb.predict_proba(Xva)[:, 1]), int((lb.coef_ != 0).sum())

# %% Step 5 — 比較｜数値だけの LR（01）より良くなるか、非ゼロの重みの数（モデルの大きさ）はどれくらいか。
out = pd.DataFrame(res).T
out["非ゼロの重み"] = pd.Series(nz)
show(out, "購入の予測（検証 day 133〜146、数値＋ハッシュ 2^20、合成データ）")
