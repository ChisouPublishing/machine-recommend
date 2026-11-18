"""第8章 8.7：表示順位の偏り。順位を無視する・順位を特徴量にして採点では 1 位に固定する・PAL 型の 3 つを、真の確率で比べる。"""
# %% Step 1 — 本書のログ｜トップのおすすめは、ロギング方策が選んだ 6 件を 1〜6 位に無作為に並べている（付録 B.4）。順位ごとのクリック率と、
#               順位の効果を除いた真のクリック確率（1 位に置いたときの確率）を並べ、差が順位だけで生じていることを確かめる。
import common  # noqa: F401  先に読み込む（LightGBM を PyTorch より先に。付録 A）

import numpy as np
import pandas as pd
import torch

from common import SEED, auc, load, show
from common.features import NUM_COLS, load_top, split
from common.models.position import PAL
from common.models.torch_ctr import DeepFM, Encoder, predict, train

torch.set_num_threads(1)   # 並列の足し合わせの順で結果が揺れないよう、1 スレッドで学習する
B = 0.15                                                       # 付録 B.3 の順位の係数（答え合わせにだけ使う）
df = load_top()
tp = load("truth_impression", truth=True).set_index("impression_id").loc[df.impression_id, "p_click"].to_numpy()
z0 = np.log(tp / (1 - tp)) + B * (df.position.to_numpy() - 1)  # 順位の効果を除いた真の logit（1 位に置いたときの値）
df = df.assign(p1=1 / (1 + np.exp(-z0)))
show(df.groupby("position").agg(クリック率=("clicked", "mean"), 真の確率_1位に置いたとき=("p1", "mean")),
     "本書のログ：順位ごとのクリック率（開発期間、合成データ）")

# %% Step 2 — 順位が商品と結び付いたログを作る｜実務のログは、前のモデルが良いと見た商品ほど上の順位に出る。ここでは前のモデルを
#               人気の方策とし、同じ表示の 6 件を傾向スコア（人気に比例）の大きい順に 1〜6 位へ並べ替え、真の式でクリックを引き直す。
prop = load("impression", columns=["impression_id", "propensity"]).set_index("impression_id").propensity
df["propensity"] = prop.loc[df.impression_id].to_numpy()
df = df.sort_values(["session_id", "propensity", "impression_id"], ascending=[True, False, True], kind="stable")
df["position"] = df.groupby("session_id").cumcount() + 1
rng = np.random.default_rng(SEED)
df["clicked"] = (rng.random(len(df)) < 1 / (1 + np.exp(-(np.log(df.p1 / (1 - df.p1)) - B * (df.position - 1))))).astype(int)
df = df.sort_values("impression_id", kind="stable")
show(df.groupby("position").agg(クリック率=("clicked", "mean"), 真の確率_1位に置いたとき=("p1", "mean")),
     "並べ替えたログ：順位ごとのクリック率（人気の順に並べた、合成データ）")

# %% Step 3 — 3 つの学び方｜(a) 順位を入れない、(b) 順位を特徴量に入れ、採点では全件を 1 位に置く、(c) PAL 型（順位の塔 × 関心の塔）。
#               モデルはどれも DeepFM（埋め込み 8 次元・2 エポック・学習率 0.002）、学習 day 28〜132、評価 day 133〜146。
tr, va = split(df, 28, 132), split(df, 133, 146)
y = tr.clicked.to_numpy().astype(float)
const = lambda d: d.assign(position=0)                         # 順位の列を 1 つの値にして、モデルに順位を見せない
enc0 = Encoder().fit(const(tr))
x0tr, x0va = enc0.transform(const(tr)), enc0.transform(const(va))
pred = {"(a) 順位を入れない": train(DeepFM(enc0.sizes(), len(NUM_COLS), k=8), x0tr, y, x0va, epochs=2, lr=2e-3, seed=SEED)}

enc = Encoder().fit(tr)
mb = DeepFM(enc.sizes(), len(NUM_COLS), k=8)
train(mb, enc.transform(tr), y, enc.transform(va), epochs=2, lr=2e-3, seed=SEED)
at = lambda r: predict(mb, enc.transform(va.assign(position=r)))
pred["(b) 順位を特徴量に、採点は 1 位"] = at(1)
ratio_b = [at(r).mean() / pred["(b) 順位を特徴量に、採点は 1 位"].mean() for r in range(1, 7)]

pal = PAL(enc0.sizes(), len(NUM_COLS), k=8)
withpos = lambda x, d: {**x, "pos": torch.from_numpy(d.position.to_numpy().astype(np.int64))}
train(pal, withpos(x0tr, tr), y, withpos(x0va, va), epochs=2, lr=2e-3, seed=SEED)
pal.relevance_only = True
s = pal.seen()
pred["(c) PAL 型"] = predict(pal, x0va) * s[0]               # 関心の塔の確率 × 1 位の見られやすさ ＝ 1 位に置いたときの確率
ratio_c = list(s / s[0])

# %% Step 4 — 答え合わせ｜評価の期間の全表示を「1 位に置いたとき」の真の確率と比べる。順位相関・平均の比のほか、人気の上位 10% の商品と
#               それ以外で平均の比を分けて見る（人気の商品ほど上に出たので、順位の効果を商品の良さと取り違えると、人気の商品を高く見積もる）。
t1 = va.p1.to_numpy()
print(f"1 位に置いたときの真のクリック確率の平均（学習 → 評価）：{tr.p1.mean():.4f} → {t1.mean():.4f}")
y1 = (np.random.default_rng(SEED + 1).random(len(va)) < t1).astype(int)   # 全件を 1 位に置いたときのクリックを真の式で引く
pop = va.propensity.to_numpy() >= np.quantile(va.propensity, 0.9)
rows = {}
for k, p in pred.items():
    rows[k] = {"AUC（1 位に置いたとき）": auc(y1, p), "真との順位相関": float(pd.Series(p).corr(pd.Series(t1), method="spearman")),
               "平均/真": p.mean() / t1.mean(), "平均/真（人気の上位 10%）": p[pop].mean() / t1[pop].mean(),
               "平均/真（それ以外）": p[~pop].mean() / t1[~pop].mean(),
               "人気の過大評価": (p[pop].mean() / t1[pop].mean()) / (p[~pop].mean() / t1[~pop].mean())}
show(pd.DataFrame(rows).T, "順位の扱い方の比較（評価 day 133〜146、真の確率は 1 位に置いたとき、合成データ）")

# %% Step 5 — 順位の効果を取り出せたか｜1 位を 1 としたときの各順位のクリック確率の比。真の値は、評価の期間の表示ごとの
#               σ(z − 0.15(r − 1)) / σ(z) の平均。
z = np.log(t1 / (1 - t1))
true_ratio = [(1 / (1 + np.exp(-(z - B * (r - 1))))).mean() / t1.mean() for r in range(1, 7)]
show(pd.DataFrame({"真の値": true_ratio, "(b) 順位を特徴量に": ratio_b, "(c) PAL 型の順位の塔": ratio_c}, index=range(1, 7)).rename_axis("順位"),
     "順位ごとのクリック確率の比（1 位 = 1、合成データ）")
