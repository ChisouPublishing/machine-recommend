"""第4章：予測確率を金額につなぐ。期待粗利で並べ替えたとき、上位の表示で実際の粗利がどれだけ取れるかを比べる。"""
# %% Step 1 — 準備｜検証期間の表示に、商品の粗利（価格 − 原価）と、カテゴリの過去の返品率を付ける。
import numpy as np
import pandas as pd

from common import connect, show
from common.data import cv_folds
from common.features import NUM_COLS, load_top, split

RETURN_COST = 1_000
df = load_top()
f = cv_folds()[-1]
tr, va = split(df, *f["train"]), split(df, *f["valid"])
con = connect()
it = con.sql("SELECT item_id, list_price - unit_cost AS margin FROM item").df()
D = "date_diff('day', DATE '2025-04-07', CAST({} AS DATE))"
rr = con.sql(f"""SELECT it.category_id, AVG((r.return_id IS NOT NULL)::INT) AS ret FROM order_line l JOIN "order" o USING (order_id)
 JOIN item it ON it.item_id = l.item_id LEFT JOIN "return" r USING (order_line_id)
 WHERE r.returned_at IS NULL OR {D.format('r.returned_at')} <= 132 GROUP BY 1""").df()
va = va.merge(it, on="item_id").merge(rr, on="category_id", how="left")
va["value"] = va.margin * (1 - va.ret) - va.ret * RETURN_COST        # 購入 1 件あたりの返品控除後の粗利（期待値）

# %% Step 2 — 実現した粗利｜実際に購入された表示について、返品を差し引いた粗利を数える（返品は記録で確かめる）。
real = con.sql("""SELECT l.impression_id, l.unit_price - l.unit_cost AS m, (r.return_id IS NOT NULL)::INT AS ret
 FROM order_line l LEFT JOIN "return" r USING (order_line_id) WHERE l.source_surface = 'top'""").df()
real["gp"] = np.where(real.ret == 1, -RETURN_COST, real.m)
va = va.merge(real[["impression_id", "gp"]], on="impression_id", how="left").fillna({"gp": 0})

# %% Step 3 — 並べ替えの規則を比べる｜購入確率の順、期待粗利（購入確率 × 1 件の粗利）の順、ランダムで、上位 k% の表示の実績を数える。
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

X = lambda d: np.log1p(d[NUM_COLS].clip(lower=0))
s = tr.sample(500_000, random_state=0)
sc = StandardScaler().fit(X(s))
p = LogisticRegression(max_iter=500).fit(sc.transform(X(s)), s.purchased).predict_proba(sc.transform(X(va)))[:, 1]
rules = {"購入確率の順": p, "期待粗利の順": p * va.value.to_numpy(), "ランダム": np.random.default_rng(0).random(len(va))}
rows = []
for name, score in rules.items():
    o = np.argsort(-score)
    for k in (0.05, 0.10, 0.20):
        top = va.iloc[o[: int(len(va) * k)]]
        rows.append((name, f"上位 {k:.0%}", top.purchased.sum(), round(top.gp.sum())))
show(pd.DataFrame(rows, columns=["並べ方", "表示", "購入数", "返品控除後の粗利（円）"]), "上位の表示の実績（検証期間、合成データ）", index=False)

# %% Step 4 — 金額のキャリブレーション｜予測した期待粗利の合計と、実現した粗利の合計を比べる。
print(f"予測した期待粗利の合計: {(p * va.value).sum():,.0f} 円")
print(f"実現した粗利の合計　　: {va.gp.sum():,.0f} 円")
