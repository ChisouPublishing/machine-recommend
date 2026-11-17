"""第3章：リークのある特徴量は、検証では強く見え、本番の条件では役に立たないことを確かめる。"""
# %% Step 1 — 準備｜特徴量の表と、リークのある 2 つの特徴量（当日の同じ商品の閲覧数・全期間の購入数）を作る。
import lightgbm as lgb
import numpy as np
import pandas as pd

from common import SEED, auc, connect, show
from common.data import cv_folds
from common.features import NUM_COLS, load_top, split

df = load_top()
con = connect()
D = "date_diff('day', DATE '2025-04-07', CAST({} AS DATE))"
f = cv_folds()[-1]                             # 開発期間の最後の折：学習 day 28〜132、検証 day 133〜146
tr, va = split(df, *f["train"]), split(df, *f["valid"])
con.register("x", pd.concat([tr, va])[["impression_id", "user_id", "item_id", "day", "shown_at"]])
leak = con.sql(f"""
SELECT x.impression_id,
  (SELECT COUNT(*) FROM event e WHERE e.user_id = x.user_id AND e.item_id = x.item_id AND e.event_type = 'view'
     AND {D.format('e.occurred_at')} = x.day) AS same_day_views,
  (SELECT COUNT(*) FROM event e WHERE e.user_id = x.user_id AND e.item_id = x.item_id AND e.event_type = 'view'
     AND {D.format('e.occurred_at')} = x.day AND e.occurred_at < x.shown_at) AS same_day_views_serving,
  (SELECT COUNT(*) FROM event e WHERE e.item_id = x.item_id AND e.event_type = 'purchase') AS item_purch_all
FROM x""").df()
tr, va = tr.merge(leak, on="impression_id"), va.merge(leak, on="impression_id")

# %% Step 2 — 学習と評価｜正しい特徴量だけのモデルと、リークを足したモデルを、同じ設定の LightGBM で比べる。
def fit(cols):
    m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31, random_state=SEED, verbose=-1)
    return m.fit(tr[cols], tr.clicked)

ok = fit(NUM_COLS)
lk = fit(NUM_COLS + ["same_day_views", "item_purch_all"])
serving = va.assign(same_day_views=va.same_day_views_serving)   # 本番：表示の時点で分かる値に置き換える
cols = NUM_COLS + ["same_day_views", "item_purch_all"]
res = pd.DataFrame([
    ("正しい特徴量", auc(va.clicked, ok.predict_proba(va[NUM_COLS])[:, 1])),
    ("リークあり（検証の条件）", auc(va.clicked, lk.predict_proba(va[cols])[:, 1])),
    ("リークあり（本番の条件）", auc(va.clicked, lk.predict_proba(serving[cols])[:, 1])),
], columns=["モデル", "検証 AUC（クリック）"])
show(res, "リークの影響（検証 day 133〜146、合成データ）", index=False)

# %% Step 3 — どの特徴量に頼ったか｜検証データで 1 列ずつ値を混ぜ、AUC がどれだけ下がるか（並べ替えの重要度）を見る。
# 木の gain は、ほぼ完全に分かれる特徴量があると後の木で数値が膨らみ当てにならないため、検証での AUC の低下で測る
base = auc(va.clicked, lk.predict_proba(va[cols])[:, 1])
rng = np.random.default_rng(SEED)
drop = {}
for c in ["same_day_views", "item_purch_all", "i_click_28d", "u_views_28d", "u_cat_views_28d"]:
    sh = va[cols].copy()
    sh[c] = rng.permutation(sh[c].to_numpy())
    drop[c] = base - auc(va.clicked, lk.predict_proba(sh)[:, 1])
show(pd.Series(drop, name="AUC の低下").sort_values(ascending=False), "リークありのモデルの並べ替えの重要度", digits=4)
