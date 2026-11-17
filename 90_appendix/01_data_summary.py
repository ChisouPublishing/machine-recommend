"""付録 B：合成データの要約（件数と、設計した性質が出ているか）。答え合わせ用の表も読む。"""
# %% Step 1 — 準備｜共通処理で DuckDB の接続を作る。truth=True は答え合わせ用の表も見せる指定。
import json

import pandas as pd

from common import DATA, connect, show

con = connect(truth=True)
meta = json.loads((DATA / "meta.json").read_text())

# %% Step 2 — 件数｜生成したテーブルの行数を並べる。
rows = {k: v["rows"] for k, v in meta["tables"].items()}
print("期間:", meta["start"], "から", meta["days"], "日、乱数の種", meta["params"]["seed"])
show(pd.Series(rows, name="行数"), "テーブルの行数")

# %% Step 3 — 推薦面ごとの率｜表示・クリック・購入の実測と、真の確率の平均を比べる。
q = """
WITH clk AS (SELECT DISTINCT impression_id FROM event WHERE event_type = 'view' AND impression_id IS NOT NULL),
     buy AS (SELECT DISTINCT impression_id FROM event WHERE event_type = 'purchase' AND impression_id IS NOT NULL)
SELECT i.surface AS 推薦面, COUNT(*) AS 表示数,
       AVG(CASE WHEN c.impression_id IS NULL THEN 0 ELSE 1 END) AS CTR,
       AVG(t.p_click) AS 真のクリック確率,
       AVG(CASE WHEN b.impression_id IS NULL THEN 0 ELSE 1 END) AS 購入率,
       AVG(t.p_purchase) AS 真の購入確率
FROM impression i JOIN truth_impression t USING (impression_id)
LEFT JOIN clk c USING (impression_id) LEFT JOIN buy b USING (impression_id)
GROUP BY 1 ORDER BY 表示数 DESC
"""
show(con.sql(q).df(), "推薦面ごとの率", index=False)

# %% Step 4 — 業務の性質｜返品率・新規の割合・クーポンの真の効果を確かめる。
q = """
SELECT (SELECT COUNT(*) FROM "return") * 1.0 / (SELECT COUNT(*) FROM order_line) AS 返品率,
       (SELECT AVG(CASE WHEN registered_at >= DATE '2025-04-07' THEN 1 ELSE 0 END) FROM "user") AS 期間中の新規会員の割合,
       (SELECT AVG(CASE WHEN release_date >= DATE '2025-04-07' THEN 1 ELSE 0 END) FROM item) AS 期間中の新商品の割合,
       (SELECT AVG(uplift) FROM truth_campaign) AS クーポンの真の効果の平均,
       (SELECT AVG(CASE WHEN uplift < 0 THEN 1 ELSE 0 END) FROM truth_campaign) AS 効果が負の割合
"""
show(con.sql(q).df().T.rename(columns={0: "値"}), "業務の性質")
