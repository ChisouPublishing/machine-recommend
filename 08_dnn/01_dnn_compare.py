"""第8章：深層学習のモデルを、最小実装（Wide & Deep・DeepFM・DIN）と FuxiCTR（Wide & Deep・DeepFM・DCN-V2・DIN・DIEN）で比べる。"""
# %% Step 1 — データ｜学習 day 28〜132（負例 10%）、評価 day 133〜146。FuxiCTR は学習の末の 7 日を早期終了に使う。
import common  # noqa: F401  先に読み込む（LightGBM を PyTorch より先に。付録 A）
import time

import pandas as pd
import torch

from common import SEED, show, summary
from common.features import NUM_COLS, load_top, split
from common.models import fuxi
from common.models.gbdt import correct, downsample
from common.models.torch_ctr import DIN, DeepFM, Encoder, WideDeep, train

torch.set_num_threads(1)   # 並列の足し合わせの順で結果が揺れないよう、1 スレッドで学習する
R = 0.1
df = load_top()
tr = downsample(split(df, 28, 132), "purchased", R, seed=SEED)
va = split(df, 133, 146)
y = va.purchased.to_numpy()
res, secs = {}, {}

# %% Step 2 — 最小実装｜埋め込み 8 次元・隠れ層 (128, 64)・3 エポック。DIN は閲覧の系列（直近 20 件の商品とカテゴリ）を使う。
enc = Encoder().fit(tr)
xtr, xva = enc.transform(tr), enc.transform(va)
for name, M in (("Wide & Deep（最小実装）", WideDeep), ("DeepFM（最小実装）", DeepFM), ("DIN（最小実装）", DIN)):
    t = time.time()
    m = M(enc.sizes(), len(NUM_COLS), k=8)
    res[name] = summary(y, correct(train(m, xtr, tr.purchased.to_numpy(), xva, epochs=3, lr=2e-3, seed=SEED), R))
    secs[name] = time.time() - t

# %% Step 3 — FuxiCTR｜同じ行で 5 つのモデル。系列は DIN・DIEN にだけ渡す（第7章の 02）。設定はモデルの既定に、次元・学習率・エポックを上書き。
cut = 126
plain = fuxi.prepare("ch08", tr[tr.day < cut], tr[tr.day >= cut], va)
seqd = fuxi.prepare("ch08_seq", tr[tr.day < cut], tr[tr.day >= cut], va, with_seq=True)
for name in ("WideDeep", "DeepFM", "DCNv2", "DIN", "DIEN"):
    t = time.time()
    data = seqd if name in ("DIN", "DIEN") else plain
    p = fuxi.fit_predict(name, data, {"embedding_dim": 8, "learning_rate": 2e-3, "epochs": 3}, seed=SEED)
    label = {"WideDeep": "Wide & Deep", "DCNv2": "DCN-V2"}.get(name, name)
    res[f"{label}（FuxiCTR）"] = summary(y, correct(p, R))
    secs[label + "（FuxiCTR）"] = time.time() - t

# %% Step 4 — 比較｜第6章の LightGBM（AUC 0.8608）・第7章の FM と比べる。学習の秒数は環境で変わるので本文に載せない。
for k, v in secs.items():
    print(f"[time] {k} {v:.1f} 秒")
show(pd.DataFrame(res).T, "深層学習のモデルの比較（評価 day 133〜146、学習は負例 10%、合成データ）")
