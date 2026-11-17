"""第6章：ランキング学習。セッションの 6 枠を 1 つのまとまり（クエリ）として、並べ方そのものを学ぶ 3 つの方法を比べる。"""
# %% Step 1 — データ｜関連度 = 購入 2・クリック 1・それ以外 0。順位（position）はログの並べ方の結果なので、特徴量から外す。
import time

import numpy as np
import pandas as pd

from common import SEED, ndcg_at_k, show
from common.data import cv_folds
from common.features import load_top, split
from common.models.gbdt import GB_COLS, frame, levels_of

df = load_top()
lv = levels_of(df)
f = cv_folds()[-1]
tr, va = split(df, *f["train"]), split(df, *f["valid"])
for d in (tr, va):
    d["rel"] = np.where(d.purchased == 1, 2, d.clicked)
COLS = [c for c in GB_COLS if c != "position"]
# 学習：関連度が 0 だけのセッションは並べ方を学ぶ材料にならないので、1 以上を含むセッションだけを使う
tr = tr[tr.groupby("session_id").rel.transform("max") > 0].sort_values(["session_id", "impression_id"])
va = va[va.groupby("session_id").rel.transform("max") > 0].sort_values(["session_id", "impression_id"])
gtr = tr.groupby("session_id", sort=False).size().to_numpy()
X = lambda d: frame(d, lv)[COLS]
print(f"学習 {len(gtr):,} セッション・{len(tr):,} 行、評価 {va.session_id.nunique():,} セッション")

# %% Step 2 — 4 つの方法｜点ごとの分類（購入かどうか）と、LambdaRank（LightGBM）・rank:ndcg（XGBoost）・YetiRank（CatBoost）。
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRanker, Pool

scores, secs = {}, {}
t = time.time()
m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, random_state=SEED, verbose=-1).fit(X(tr), tr.purchased)
scores["点ごとの分類（LightGBM）"], secs["点ごと"] = m.predict_proba(X(va))[:, 1], time.time() - t
t = time.time()
m = lgb.LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.05, num_leaves=63, random_state=SEED,
                   verbose=-1).fit(X(tr), tr.rel, group=gtr)
scores["LambdaRank（LightGBM）"], secs["LambdaRank"] = m.predict(X(va)), time.time() - t
t = time.time()
m = xgb.XGBRanker(objective="rank:ndcg", tree_method="hist", enable_categorical=True, n_estimators=300, learning_rate=0.05,
                  max_depth=6, random_state=SEED).fit(X(tr), tr.rel, group=gtr)
scores["rank:ndcg（XGBoost）"], secs["rank:ndcg"] = m.predict(X(va)), time.time() - t
t = time.time()
cats = [COLS.index(c) for c in COLS if c in lv]
to_s = lambda d: X(d).astype({c: str for c in lv})
m = CatBoostRanker(loss_function="YetiRank", iterations=300, learning_rate=0.05, depth=6, random_seed=SEED, verbose=False,
                   allow_writing_files=False)
m.fit(Pool(to_s(tr), tr.rel.to_numpy(), group_id=tr.session_id.to_numpy(), cat_features=cats))
scores["YetiRank（CatBoost）"] = m.predict(Pool(to_s(va), group_id=va.session_id.to_numpy(), cat_features=cats))
secs["YetiRank"] = time.time() - t
for k, v in secs.items():
    print(f"[time] {k} {v:.1f} 秒")

# %% Step 3 — 評価｜セッションごとに点数の順に並べ、NDCG@3・NDCG@6 と、1 位に購入された商品が来た割合を出す。ログの順位とも比べる。
scores["ログの順位（人気＋ε）"] = -va.position.to_numpy()
rows = []
for name, sc in scores.items():
    v = va.assign(s=sc)
    lists = [g.sort_values("s", ascending=False).rel.to_numpy() for _, g in v.groupby("session_id", sort=False)]
    buy = [l for l in lists if (l == 2).any()]
    rows.append((name, ndcg_at_k(lists, 3), ndcg_at_k(lists, 6), np.mean([l[0] == 2 for l in buy])))
show(pd.DataFrame(rows, columns=["方法", "NDCG@3", "NDCG@6", "購入が 1 位の割合"]),
     "ランキングの比較（評価 day 133〜146、関連度 1 以上を含むセッション、合成データ）", index=False)
