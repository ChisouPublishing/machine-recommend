"""第5章：採用したロジスティック回帰を、係数・感度・真の確率との比較で読む。ベースラインとして記録を残す。"""
# %% Step 1 — 学習｜第5章で採用した構成（C と bits）で、開発期間の最後の折の学習期間に当てはめる。
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common import OUT, connect, savefig, show, summary
from common.data import cv_folds
from common.features import NUM_COLS, load_top, split
from common.models.linear import NumScaler, design, hash_codes

cfg = json.loads((OUT / "05_lr" / "best_lr.json").read_text())
df = load_top()
f = cv_folds()[-1]
tr, va = split(df, *f["train"]), split(df, *f["valid"])
sc = NumScaler().fit(tr)
m = LogisticRegression(C=cfg["C"], max_iter=300).fit(design(tr, sc, hash_codes(tr), cfg["bits"]), tr.purchased)
p = m.predict_proba(design(va, sc, hash_codes(va), cfg["bits"]))[:, 1]
show(pd.Series(summary(va.purchased, p), name="値"), f"採用した LR（C={cfg['C']:.4f}, bits={cfg['bits']}）の検証の指標")

# %% Step 2 — 係数｜数値の特徴量は標準化してあるので、係数の大きさを比べられる（1 標準偏差あたりの対数オッズの変化）。
coef = pd.Series(m.coef_[0][: len(NUM_COLS)], index=NUM_COLS).sort_values()
show(pd.concat([coef.head(4), coef.tail(5)]).rename("係数"), "数値の特徴量の係数（下位 4・上位 5）", digits=3)

# %% Step 3 — 感度｜検証の行の値を 1 つだけ動かしたとき、平均の予測確率がどう変わるか（他の列はそのまま）。
def avg_with(col, value):
    v = va.copy()
    v[col] = value
    return m.predict_proba(design(v, sc, hash_codes(v), cfg["bits"]))[:, 1].mean()

rows = [("順位", k, avg_with("position", k)) for k in (1, 3, 6)]
rows += [("会員のサイズの在庫", k, avg_with("size_in_stock", k)) for k in (0, 1)]
show(pd.DataFrame(rows, columns=["動かした列", "値", "平均の予測確率"]), "感度（検証期間の全行の平均）", index=False, digits=4)

# %% Step 4 — 真の確率との比較｜答え合わせ用の真の購入確率と並べ、予測確率の偏りを区間ごとに見る。
t = connect(truth=True).sql("SELECT impression_id, p_purchase FROM truth_impression").df()
pt = va[["impression_id"]].merge(t, on="impression_id").p_purchase.to_numpy()
o = np.argsort(p)
bins = np.array_split(o, 15)
print(f"真の確率との相関 {np.corrcoef(p, pt)[0, 1]:.3f}、区間ごとの |予測 − 真の確率| の平均 "
      f"{np.mean([abs(p[i].mean() - pt[i].mean()) for i in bins]):.4f}")
fig, ax = plt.subplots(figsize=(5.2, 4.6))
ax.plot([pt[i].mean() for i in bins], [p[i].mean() for i in bins], marker="o", ms=3, label="区間ごとの平均")
ax.plot([1e-4, 0.5], [1e-4, 0.5], color="gray", lw=0.8, ls="--")
ax.set(xscale="log", yscale="log", xlim=(5e-4, 0.5), ylim=(5e-4, 0.5), xlabel="真の購入確率（対数）", ylabel="LR の予測確率（対数）")
ax.legend(fontsize=8)
savefig(fig, "05_lr", "lr_vs_truth")
