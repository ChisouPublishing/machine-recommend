"""第1章：返品控除後の粗利を頂点に、業務 KPI を分解する（開発期間の 17 週）。"""
# %% Step 1 — 準備｜共通処理の接続と、仮置きの費用を決める。費用は業務で決める値で、ここでは説明のための仮置き。
import pandas as pd

from common import PERIODS, connect, save_table, show

RETURN_COST = 1_000          # 返品 1 件の処理費（送料・検品。仮置き）
COUPON_RATE = 0.10           # クーポンの値引き率（campaign.discount_rate と同じ）
a, b = PERIODS["dev"]
con = connect()
D = "date_diff('day', DATE '2025-04-07', CAST({} AS DATE))"

# %% Step 2 — 来訪と注文｜会員数（UV）・セッション数・注文数・売上・原価を、開発期間で集計する。
base = con.sql(f"""
WITH s AS (SELECT * FROM session WHERE {D.format('started_at')} BETWEEN {a} AND {b}),
o AS (SELECT * FROM "order" WHERE {D.format('ordered_at')} BETWEEN {a} AND {b}),
l AS (SELECT l.* FROM order_line l JOIN o USING (order_id))
SELECT (SELECT COUNT(DISTINCT user_id) FROM s) AS uv,
       (SELECT COUNT(*) FROM s) AS sessions,
       (SELECT COUNT(*) FROM o) AS orders,
       (SELECT SUM(unit_price * quantity) FROM l) AS sales,
       (SELECT SUM(unit_cost * quantity) FROM l) AS cost,
       (SELECT COUNT(*) FROM l JOIN "return" r USING (order_line_id)) AS returns,
       (SELECT SUM(l.unit_price - l.unit_cost) FROM l JOIN "return" r USING (order_line_id)) AS returned_margin,
       (SELECT COUNT(*) FROM l) AS lines
""").df().iloc[0]

# %% Step 3 — クーポンの費用｜開発期間に配った 1 回目のクーポンの利用数 × 平均注文額 × 値引き率で見積もる。
used = con.sql(f"""SELECT COUNT(*) FROM campaign_assignment
                   WHERE coupon_used AND {D.format('sent_at')} BETWEEN {a} AND {b}""").fetchall()[0][0]
aov = base.sales / base.orders
coupon_cost = used * aov * COUPON_RATE

# %% Step 4 — KPI ツリー｜頂点の「返品控除後の粗利」から、来訪・転換・客単価・返品に分解して並べる。
gp = base.sales - base.cost
gp_net = gp - base.returned_margin - base.returns * RETURN_COST - coupon_cost
tree = pd.DataFrame([
    ("返品控除後の粗利（円）", gp_net, ""),
    ("├ 粗利（円）", gp, "売上 − 原価"),
    ("│ ├ 売上（円）", base.sales, "UV × 来訪頻度 × CVR × 客単価"),
    ("│ │ ├ UV（会員数）", base.uv, ""),
    ("│ │ ├ 来訪頻度（回/人）", base.sessions / base.uv, "セッション ÷ UV"),
    ("│ │ ├ CVR（注文/セッション）", base.orders / base.sessions, ""),
    ("│ │ └ 客単価（円/注文）", aov, ""),
    ("│ └ 粗利率", gp / base.sales, ""),
    ("├ 返品で失う粗利（円）", -base.returned_margin, "返品された明細の粗利"),
    ("│ └ 返品率（明細）", base.returns / base.lines, ""),
    ("├ 返品の処理費（円）", -base.returns * RETURN_COST, f"返品 1 件 {RETURN_COST:,} 円（仮置き）"),
    ("└ クーポンの値引き（円）", -coupon_cost, f"利用 {used:,} 件 × 客単価 × {COUPON_RATE:.0%}"),
], columns=["KPI", "値", "内訳・定義"])
tree["値"] = tree["値"].map(lambda v: f"{v:,.0f}" if abs(v) >= 100 else f"{v:.4f}")
print(f"開発期間 day {a}〜{b}（17 週）。合成データでの集計")
show(tree, index=False, left=("KPI", "内訳・定義"))
save_table(tree, "01_business", "kpi_tree")
