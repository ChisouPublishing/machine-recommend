"""第1章：推薦面ごとに、表示・クリック・購入・返品・粗利を並べ、推薦がどこに効いているかを見る。"""
# %% Step 1 — 準備｜共通処理の接続を作り、開発期間に絞る。
from common import PERIODS, connect, save_table, show

RETURN_COST = 1_000
a, b = PERIODS["dev"]
con = connect()
D = "date_diff('day', DATE '2025-04-07', CAST({} AS DATE))"

# %% Step 2 — 表示とクリック｜推薦面ごとの表示数と CTR。
imp = con.sql(f"""
WITH clk AS (SELECT DISTINCT impression_id FROM event WHERE event_type = 'view' AND impression_id IS NOT NULL)
SELECT surface AS 推薦面, COUNT(*) AS 表示数, AVG((c.impression_id IS NOT NULL)::INT) AS CTR
FROM impression i LEFT JOIN clk c USING (impression_id)
WHERE {D.format('shown_at')} BETWEEN {a} AND {b} GROUP BY 1
""").df()

# %% Step 3 — 購入と返品｜注文明細の経路（source_surface）ごとに、明細数・売上・返品率・返品控除後の粗利を出す。
sales = con.sql(f"""
SELECT l.source_surface AS 推薦面, COUNT(*) AS 明細数, SUM(l.unit_price) AS 売上,
       AVG((r.return_id IS NOT NULL)::INT) AS 返品率,
       SUM(l.unit_price - l.unit_cost) - SUM(CASE WHEN r.return_id IS NOT NULL
           THEN l.unit_price - l.unit_cost + {RETURN_COST} ELSE 0 END) AS 返品控除後の粗利
FROM order_line l JOIN "order" o USING (order_id) LEFT JOIN "return" r USING (order_line_id)
WHERE {D.format('o.ordered_at')} BETWEEN {a} AND {b} GROUP BY 1
""").df()

# %% Step 4 — まとめ｜推薦面（top・related・cart・mail）と、推薦を通らない経路（browse・search）を 1 つの表にする。
t = sales.merge(imp, on="推薦面", how="left")
t["表示あたり粗利（円）"] = t["返品控除後の粗利"] / t["表示数"]
t["粗利の構成比"] = t["返品控除後の粗利"] / t["返品控除後の粗利"].sum()
t = t.sort_values("返品控除後の粗利", ascending=False)
t["表示数"] = t["表示数"].astype("Int64")
t["返品控除後の粗利"] = t["返品控除後の粗利"].round().astype("int64")
cols = ["推薦面", "表示数", "CTR", "明細数", "返品率", "返品控除後の粗利", "粗利の構成比", "表示あたり粗利（円）"]
show(t[cols], "推薦面ごとの成果（開発期間、合成データ）", index=False)
save_table(t[cols], "01_business", "surface")
