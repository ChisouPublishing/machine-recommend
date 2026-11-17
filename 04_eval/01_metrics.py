"""第4章：予測確率の指標（AUC・LogLoss・PR-AUC・ECE）を、2 つの簡単なモデルと「真の確率」で比べる。"""
# %% Step 1 — 準備｜開発期間の最後の折（学習 day 28〜132、検証 day 133〜146）で、購入（表示から購入）を予測する。
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import SEED, connect, savefig, show, summary
from common.data import cv_folds
from common.features import NUM_COLS, load_top, split

df = load_top()
f = cv_folds()[-1]
tr, va = split(df, *f["train"]), split(df, *f["valid"])
y = va.purchased.to_numpy()

# %% Step 2 — 人気の率｜商品ごとの学習期間の購入率（件数の少ない商品は全体の率に寄せる）。会員を見ない基準線。
prior, k = tr.purchased.mean(), 200
g = tr.groupby("item_id").purchased.agg(["sum", "count"])
rate = (g["sum"] + prior * k) / (g["count"] + k)
p_pop = va.item_id.map(rate).fillna(prior).to_numpy()

# %% Step 3 — ロジスティック回帰｜数値の特徴量だけの簡単なモデル（第5章で詳しく扱う）。学習は 50 万行に抜き出す。
X = lambda d: np.log1p(d[NUM_COLS].clip(lower=0))
s = tr.sample(500_000, random_state=SEED)
sc = StandardScaler().fit(X(s))
lr = LogisticRegression(max_iter=500).fit(sc.transform(X(s)), s.purchased)
p_lr = lr.predict_proba(sc.transform(X(va)))[:, 1]

# %% Step 4 — 真の確率｜答え合わせ用の表から、表示ごとの真の購入確率を読む。どのモデルもこれを超えられない。
t = connect(truth=True).sql("SELECT impression_id, p_purchase FROM truth_impression").df()
p_true = va[["impression_id"]].merge(t, on="impression_id").p_purchase.to_numpy()

# %% Step 5 — 指標を並べる｜「LR の確率を 2 倍にしたもの」も加え、順位の指標と確率の指標の違いを見る。
cand = {"人気の率": p_pop, "ロジスティック回帰": p_lr, "LR の確率 × 2": np.clip(p_lr * 2, 0, 1), "真の確率": p_true}
res = pd.DataFrame({k: summary(y, p) for k, p in cand.items()}).T
show(res, "購入の予測の指標（検証 day 133〜146、合成データ）")

# %% Step 6 — キャリブレーション曲線｜予測確率を 15 区間に分け、区間ごとの平均予測と実測の購入率を対数の軸で描く。
fig, ax = plt.subplots(figsize=(5.2, 4.6))
for name in ["人気の率", "ロジスティック回帰", "LR の確率 × 2", "真の確率"]:
    p = cand[name]
    o = np.argsort(p)
    b = np.array_split(o, 15)
    ax.plot([p[i].mean() for i in b], [y[i].mean() for i in b], marker="o", ms=3, label=name)
ax.plot([1e-4, 0.5], [1e-4, 0.5], color="gray", lw=0.8, ls="--")
ax.set(xscale="log", yscale="log", xlim=(5e-4, 0.5), ylim=(5e-4, 0.5),
       xlabel="平均の予測確率（対数）", ylabel="実測の購入率（対数）")
ax.legend(fontsize=8)
savefig(fig, "04_eval", "calibration")
