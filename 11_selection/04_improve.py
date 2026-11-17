"""第11章：改善サイクル。誤差の分析 → 特徴量の追加・混ぜる → 優先順位。データの性質を変えた感度分析と、実データでの再現も行う。"""
# %% Step 1 — 誤差の分析｜ベンチマークの最良（LightGBM、種 0）の評価の予測を、層ごとに LogLoss と「平均の予測 − 実測」で見る。
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from common import CODE, OUT, SEED, auc, logloss, show
from common.bench import PERIOD, calibrate, rest_of
from common.features import load_top
from common.models.gbdt import correct, downsample, fit_predict, frame, levels_of

df = load_top()
df = df[df.day <= 146]
lv = levels_of(df)
rest = rest_of(df)
ev = rest[rest.day >= PERIOD["eval"][0]].copy()
ev["p"] = np.load(OUT / "11_selection" / "bench" / "LightGBM_0.npz")["p"]
seg = {"新規会員（登録 30 日以内）": ev.days_since_reg < 30, "新商品（発売 14 日以内）": ev.days_since_release < 14,
       "サイズの在庫なし": ev.size_in_stock == 0, "閲覧の系列が短い（5 件未満）": ev.hist_items.str.count(r"\^") < 4,
       "その他": None}
rows = []
mask_any = np.zeros(len(ev), bool)
for name, m in seg.items():
    m = ~mask_any if m is None else m.to_numpy()
    mask_any |= m
    s = ev[m]
    rows.append((name, len(s), s.purchased.mean(), s.p.mean() - s.purchased.mean(), logloss(s.purchased, s.p)))
show(pd.DataFrame(rows, columns=["層", "行数", "実測の購入率", "平均の予測 − 実測", "LogLoss"]), "層ごとの誤差（LightGBM、評価 day 133〜146）",
     index=False, digits=4)

# %% Step 2 — 特徴量を足す｜閲覧の系列のうち、表示した商品と同じカテゴリの件数（hist_same_cat）。ER の event と item から作れる。
for d in (df, rest, ev):
    d["hist_same_cat"] = [h.split("^").count(str(c)) if h else 0 for h, c in zip(d.hist_cats, d.category_id)]
import json

cfg = json.loads((OUT / "06_gbdt" / "best_gbdt.json").read_text())["lgbm"]
tr = downsample(df[(df.day >= PERIOD["train"][0]) & (df.day <= PERIOD["train"][1])], "purchased", 0.1, seed=SEED)


def lgbm(extra):
    X = lambda d: pd.concat([frame(d, lv), d[extra]], axis=1) if extra else frame(d, lv)
    q, _ = fit_predict("lgbm", cfg, X(tr), tr.purchased.to_numpy(), X(rest), seed=SEED)
    return calibrate(correct(q, 0.1), rest)


p_base, p_add = lgbm([]), lgbm(["hist_same_cat"])
din = np.load(OUT / "11_selection" / "bench" / "DIN（最小実装）_0.npz")["p"]
lg = lambda p: np.log(p / (1 - p))
blend = 1 / (1 + np.exp(-(0.5 * lg(p_base) + 0.5 * lg(din))))
y = ev.purchased.to_numpy()
show(pd.DataFrame({"LogLoss": {k: logloss(y, p) for k, p in [("LightGBM（ベンチマーク）", p_base), ("＋同じカテゴリの閲覧数", p_add),
                                                        ("LightGBM と DIN を半分ずつ混ぜる", blend)]},
                   "AUC": {k: auc(y, p) for k, p in [("LightGBM（ベンチマーク）", p_base), ("＋同じカテゴリの閲覧数", p_add),
                                                ("LightGBM と DIN を半分ずつ混ぜる", blend)]}}),
     "改善の候補（評価 day 133〜146、種 0、合成データ）", digits=5)

# %% Step 3 — 感度分析｜系列の効果を弱めたデータ（--seq-strength 0.3）を作り、LR・LightGBM・DIN の順位が変わるかを確かめる。
weak = CODE / "data" / "generated" / "weak_seq"
if not (weak / "meta.json").exists():
    subprocess.run([sys.executable, "data/make_data.py", "--seq-strength", "0.3", "--out", str(weak)], cwd=CODE, check=True,
                   capture_output=True)
for d in (CODE / "data" / "generated" / "default", weak):
    r = subprocess.run([sys.executable, "11_selection/sensitivity_one.py"], cwd=CODE, capture_output=True, text=True,
                       env=dict(os.environ, SUISEN_DATA=str(d), PYTHONPATH=str(CODE)))
    print(r.stdout.strip())
