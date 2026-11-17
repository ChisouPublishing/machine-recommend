"""第5章：numpy で書いたロジスティック回帰（最小実装）と、scikit-learn の LogisticRegression を同じ目的関数で比べる。"""
# %% Step 1 — データ｜開発期間の最後の折（学習 day 28〜132、検証 day 133〜146）。ラベルは表示からの購入。数値の特徴量 20 列だけを使う。
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common import show, summary
from common.data import cv_folds
from common.features import NUM_COLS, load_top, split
from common.models.linear import NumpyLR, NumScaler

df = load_top()
f = cv_folds()[-1]
tr, va = split(df, *f["train"]), split(df, *f["valid"])
sc = NumScaler().fit(tr)
Xtr, Xva, ytr, yva = sc.transform(tr), sc.transform(va), tr.purchased.to_numpy(), va.purchased.to_numpy()
L2 = 1e-4

# %% Step 2 — 最小実装｜全件の勾配で 1,500 回更新する。
t = time.time()
mine = NumpyLR(l2=L2, lr=0.05, epochs=1500).fit(Xtr, ytr)
print(f"[time] numpy の LR {time.time() - t:.1f} 秒")

# %% Step 3 — ライブラリ｜scikit-learn の LogisticRegression。正則化の強さ C を、同じ目的関数になるよう C = 1/(λ·n) にそろえる。
t = time.time()
lib = LogisticRegression(C=1 / (L2 * len(ytr)), max_iter=1000).fit(Xtr, ytr)
print(f"[time] scikit-learn の LR {time.time() - t:.1f} 秒")

# %% Step 4 — 比較｜検証の指標・目的関数の値・係数の最大の差。同じ目的関数の最小点なので、指標と目的関数は一致するはず。
def objective(w, b):
    p = np.clip(1 / (1 + np.exp(-(Xtr @ w + b))), 1e-12, 1 - 1e-12)
    return -(ytr * np.log(p) + (1 - ytr) * np.log(1 - p)).mean() + L2 / 2 * (w @ w)



res = pd.DataFrame({"numpy（最小実装）": summary(yva, mine.predict_proba(Xva)),
                    "scikit-learn": summary(yva, lib.predict_proba(Xva)[:, 1])}).T
show(res, "購入の予測（検証 day 133〜146、数値の特徴量 20 列、合成データ）")
print(f"学習の目的関数: numpy {objective(mine.w, mine.b):.7f} ／ scikit-learn {objective(lib.coef_[0], lib.intercept_[0]):.7f}")
print("係数の最大の差:", f"{np.abs(mine.w - lib.coef_[0]).max():.5f}", "／切片の差:", f"{abs(mine.b - lib.intercept_[0]):.5f}")
