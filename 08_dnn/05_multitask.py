"""第8章 8.6：クリックと購入を 1 つのモデルで学ぶ。別々のモデル・Shared-Bottom・ESMM・MMoE（最小実装）を、真の確率で答え合わせする。"""
# %% Step 1 — データ｜学習 day 28〜132 の全表示（間引かない）、評価 day 133〜146。答え合わせ用に、表示ごとの真の CTR・CVR・購入確率を読む。
import common  # noqa: F401  先に読み込む（LightGBM を PyTorch より先に。付録 A）
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from common import SEED, auc, load, logloss, savefig, show
from common.features import NUM_COLS, load_top, split
from common.models import multitask as mt
from common.models.torch_ctr import DeepFM, Encoder, train

torch.set_num_threads(1)   # 並列の足し合わせの順で結果が揺れないよう、1 スレッドで学習する
df = load_top()
tr, va = split(df, 28, 132), split(df, 133, 146)
truth = load("truth_impression", truth=True).set_index("impression_id")
tv = truth.loc[va.impression_id]
click, buy = tr.clicked.to_numpy(), tr.purchased.to_numpy()
print(f"学習 {len(tr):,} 表示（クリック {click.sum():,}、購入 {buy.sum():,}）、評価 {len(va):,} 表示")
tt_ = truth.loc[tr.impression_id]
print("真の平均（学習 → 評価）：" + "、".join(f"{k} {tt_[c].mean():.4f} → {tv[c].mean():.4f}" for k, c in
      (("CTR", "p_click"), ("クリック後の CVR", "p_purchase_given_click"), ("表示 → 購入", "p_purchase"))))
enc = Encoder().fit(tr)
xtr, xva = enc.transform(tr), enc.transform(va)
clk = tr.clicked.to_numpy() == 1
xtr_c = {k: v[torch.from_numpy(clk)] for k, v in xtr.items()}          # クリックした表示だけ（CVR を従来の作り方で学ぶ行）
preds, secs = {}, {}

# %% Step 2 — 別々のモデル｜DeepFM（第8章 8.2 の最小実装）を 3 つ：CTR（全表示）、CVR（クリックした表示だけ）、表示 → 購入（全表示）。
t = time.time()
m = lambda: DeepFM(enc.sizes(), len(NUM_COLS), k=8)
p_ctr = train(m(), xtr, click.astype(float), xva, epochs=2, lr=2e-3, seed=SEED)
p_cvr = train(m(), xtr_c, buy[clk].astype(float), xva, epochs=2, lr=2e-3, seed=SEED)
preds["別々（CTR × CVR）"] = (p_ctr, p_cvr)
p_direct = train(m(), xtr, buy.astype(float), xva, epochs=2, lr=2e-3, seed=SEED)
preds["購入を直接（参考）"] = (None, None, p_direct)
secs["別々のモデル（3 つ）"] = time.time() - t

# %% Step 3 — 1 つのモデルで 2 つのタスク｜3 つの構造それぞれを、2 つの損失（CVR はクリックした表示だけ／ESMM の損失）で学ぶ。
#               構造の差と損失の差を分けて見るため。埋め込み 8 次元・2 エポック・学習率 0.002 はすべて同じ。
for sname, M in (("Shared-Bottom", mt.SharedBottom), ("埋め込み共有", mt.ESMM), ("MMoE", mt.MMoE)):
    for lname, mode in (("クリックのみ", "click_only"), ("ESMM の損失", "esmm")):
        name = f"{sname}・{lname}"
        t = time.time()
        model = mt.train(M(enc.sizes(), len(NUM_COLS), k=8), xtr, click, buy, mode, epochs=2, lr=2e-3, seed=SEED)
        preds[name] = mt.predict(model, xva)
        secs[name] = time.time() - t

# %% Step 4 — 答え合わせ｜CTR の AUC、CVR は「全表示」での真の CVR との順位相関と平均の比（クリックしなかった表示にも予測を出すため）、
#               表示 → 購入は実際の購入の AUC・LogLoss と、真の購入確率との順位相関。
corr = lambda a, b: float(pd.Series(a).corr(pd.Series(b), method="spearman"))
yc, yb = va.clicked.to_numpy(), va.purchased.to_numpy()
t_ctr, t_cvr, t_buy = tv.p_click.to_numpy(), tv.p_purchase_given_click.to_numpy(), tv.p_purchase.to_numpy()
nc = yc == 0
rows = {}
for name, pr in preds.items():
    r = {}
    if pr[0] is not None:
        pc, pv = pr
        pb = pc * pv
        r.update({"CTR AUC": auc(yc, pc), "CVR 真との相関": corr(pv, t_cvr), "CVR 平均/真（未クリック）": pv[nc].mean() / t_cvr[nc].mean()})
    else:
        pb = pr[2]
    r.update({"購入 AUC": auc(yb, pb), "購入 LogLoss": logloss(yb, pb), "購入 真との相関": corr(pb, t_buy),
              "購入 平均/真": pb.mean() / t_buy.mean()})
    rows[name] = r
for k, v in secs.items():
    print(f"[time] {k} {v:.1f} 秒")
show(pd.DataFrame(rows).T, "クリックと購入を同時に学ぶ（評価 day 133〜146、真の確率で答え合わせ、合成データ）")

# %% Step 5 — 図｜クリックしなかった表示を、真の CVR の大きさで 10 等分し、区間ごとの予測の平均を真の平均と並べる。
q = pd.qcut(t_cvr[nc], 10, labels=False, duplicates="drop")
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(pd.Series(t_cvr[nc]).groupby(q).mean(), pd.Series(t_cvr[nc]).groupby(q).mean(), "k--", label="真の CVR（理想）")
for name, mk in (("別々（CTR × CVR）", "o"), ("埋め込み共有・クリックのみ", "s"), ("埋め込み共有・ESMM の損失", "^")):
    ax.plot(pd.Series(t_cvr[nc]).groupby(q).mean(), pd.Series(preds[name][1][nc]).groupby(q).mean(), marker=mk, label=name)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("真の CVR（クリックしなかった表示、10 区間の平均）"); ax.set_ylabel("予測した CVR の平均")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
savefig(fig, "08_dnn", "multitask_cvr")
