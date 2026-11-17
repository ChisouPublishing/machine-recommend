"""第2章：ER 図の表ごとに、件数・主キーの一意性・外部キーの欠け・1 対多の多さを確かめる。"""
# %% Step 1 — 準備｜ER 図の主キーと、確かめる 1 対多の関係（親の表・子の表・結合キー）を並べる。
import pandas as pd

from common import connect, show

con = connect()
PK = {"user": "user_id", "item": "item_id", "sku": "sku_id", "brand": "brand_id", "category": "category_id",
      "session": "session_id", "impression": "impression_id", "event": "event_id", '"order"': "order_id",
      "order_line": "order_line_id", '"return"': "return_id", "campaign_assignment": "assignment_id"}
REL = [("user", "session", "user_id"), ("session", "impression", "session_id"), ("session", "event", "session_id"),
       ("item", "sku", "item_id"), ("sku", "order_line", "sku_id"), ('"order"', "order_line", "order_id"),
       ("order_line", '"return"', "order_line_id"), ("item", "impression", "item_id")]

# %% Step 2 — 主キー｜件数と、主キーが一意かを数える。
rows = []
for t, k in PK.items():
    n, u = con.sql(f"SELECT COUNT(*), COUNT(DISTINCT {k}) FROM {t}").fetchall()[0]
    rows.append((t.strip('"'), n, n == u))
show(pd.DataFrame(rows, columns=["表", "件数", "主キーが一意"]), "主キー", index=False)

# %% Step 3 — 1 対多｜子の表の外部キーが欠けた件数・親に無い件数と、親 1 件あたりの子の件数を出す。
rows = []
for parent, child, k in REL:
    null, orphan, mean, p99 = con.sql(f"""
    WITH c AS (SELECT {k} FROM {child}),
         per AS (SELECT p.{k}, COUNT(c.{k}) AS n FROM {parent} p LEFT JOIN c USING ({k}) GROUP BY ALL)
    SELECT (SELECT COUNT(*) FROM c WHERE {k} IS NULL),
           (SELECT COUNT(*) FROM c WHERE {k} IS NOT NULL AND {k} NOT IN (SELECT {k} FROM {parent})),
           (SELECT AVG(n) FROM per), (SELECT quantile_cont(n, 0.99) FROM per)
    """).fetchall()[0]
    rows.append((f"{parent.strip('\"')} → {child.strip('\"')}", null, orphan, mean, p99))
show(pd.DataFrame(rows, columns=["関係（1 → 多）", "外部キーが空", "親に無い", "親 1 件あたり平均", "99% 点"]),
     "1 対多の関係", index=False, digits=2)
