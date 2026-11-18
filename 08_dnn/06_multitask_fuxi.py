"""第8章 8.6 の発展版：FuxiCTR の多タスクのモデル（ShareBottom・MMoE・PLE）で、クリックと購入を同時に学ぶ。"""
# %% Step 1 — データ｜学習 day 28〜125、早期終了 day 126〜132、評価 day 133〜146（間引かない）。正解は 2 列（クリック・購入）。
import common  # noqa: F401  先に読み込む（LightGBM を PyTorch より先に。付録 A）
import time

import pandas as pd

from common import SEED, auc, load, logloss, show
from common.features import load_top, split
from common.models import fuxi

df = load_top()
tr, ho, va = split(df, 28, 125), split(df, 126, 132), split(df, 133, 146)
tv = load("truth_impression", truth=True).set_index("impression_id").loc[va.impression_id]
data = fuxi.prepare("ch08_mtl", tr, ho, va, label=["clicked", "purchased"])

# %% Step 2 — 3 つのモデル｜FuxiCTR の多タスクのモデルは、2 つの正解をそれぞれ全表示で学ぶ（タスク 1：表示 → クリック、
#               タスク 2：表示 → 購入）。CVR の塔は無いので、最小実装の ESMM とは表示 → 購入で比べる。設定は既定に、次元・学習率・
#               エポック・層の大きさを上書きして、最小実装と同じくらいの大きさにする。
over = {"embedding_dim": 8, "learning_rate": 2e-3, "epochs": 2, "net_dropout": 0.1, "tower_hidden_units": [64],
        "bottom_hidden_units": [128], "expert_hidden_units": [128], "gate_hidden_units": [], "num_experts": 4,
        "num_shared_experts": 2, "num_specific_experts": 2, "num_layers": 1}
rows, secs = {}, {}
corr = lambda a, b: float(pd.Series(a).corr(pd.Series(b), method="spearman"))
yc, yb = va.clicked.to_numpy(), va.purchased.to_numpy()
for name in ("ShareBottom", "MMoE", "PLE"):
    t = time.time()
    p = fuxi.fit_predict(name, data, over, seed=SEED)
    secs[name] = time.time() - t
    rows[f"{name}（FuxiCTR）"] = {"CTR AUC": auc(yc, p["clicked"]), "購入 AUC": auc(yb, p["purchased"]),
                                 "購入 LogLoss": logloss(yb, p["purchased"]), "購入 真との相関": corr(p["purchased"], tv.p_purchase.to_numpy()),
                                 "購入 平均/真": p["purchased"].mean() / tv.p_purchase.mean()}

# %% Step 3 — 比較｜最小実装の結果（05 の表）と並べて読む。学習の秒数は環境で変わるので本文に載せない。
for k, v in secs.items():
    print(f"[time] {k} {v:.1f} 秒")
show(pd.DataFrame(rows).T, "FuxiCTR の多タスクのモデル（評価 day 133〜146、真の確率で答え合わせ、合成データ）")
