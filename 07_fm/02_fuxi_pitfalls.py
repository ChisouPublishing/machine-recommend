"""第7章：ライブラリ（FuxiCTR）の FM が、渡し方しだいで学習できないことを確かめる（数値の列の扱い・系列の扱い）。"""
# %% Step 1 — データ｜学習 day 28〜125（負例 10%）、早期終了用 day 126〜132、評価 day 133〜146。
import common  # noqa: F401  先に読み込む（LightGBM を PyTorch より先に。付録 A）
import pandas as pd
import torch

from common import SEED, show, summary
from common.features import load_top, split
from common.models import fuxi
from common.models.gbdt import correct, downsample

torch.set_num_threads(1)   # 並列の足し合わせの順で結果が揺れないよう、1 スレッドで学習する
R = 0.1
df = load_top()
ftr, fho = downsample(split(df, 28, 125), "purchased", R, seed=SEED), downsample(split(df, 126, 132), "purchased", R, seed=SEED)
va = split(df, 133, 146)
y = va.purchased.to_numpy()
cfg = {"embedding_dim": 8, "learning_rate": 5e-3, "epochs": 3}

# %% Step 2 — 3 つの渡し方｜(a) 数値のまま＋系列あり、(b) 数値を区間に分ける＋系列あり、(c) 数値を区間に分ける＋系列なし。
cases = {"(a) 数値のまま・系列あり": dict(numeric="raw", with_seq=True),
         "(b) 数値を区間に・系列あり": dict(numeric="bin", with_seq=True),
         "(c) 数値を区間に・系列なし": dict(numeric="bin", with_seq=False)}
res = {}
for name, kw in cases.items():
    data = fuxi.prepare("ch07_" + kw["numeric"] + ("_seq" if kw["with_seq"] else ""), ftr, fho, va, **kw)
    res[name] = summary(y, correct(fuxi.fit_predict("FM", data, cfg, seed=SEED), R))

# %% Step 3 — 比較｜同じ FM・同じ学習の設定で、入力の渡し方だけが違う。
show(pd.DataFrame(res).T, "FuxiCTR の FM と入力の渡し方（評価 day 133〜146、合成データ）")
