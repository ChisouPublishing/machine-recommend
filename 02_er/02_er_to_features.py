"""第2章：ER の 1 対多を集約すると特徴量になる。ID の種類の多さと人気の偏りも確かめる。"""
# %% Step 1 — 準備｜開発期間の閲覧を、商品の表と結合して読む（event → item は多対 1）。
import numpy as np

from common import PERIODS, connect, show

con = connect()
a, b = PERIODS["dev"]
D = "date_diff('day', DATE '2025-04-07', CAST({} AS DATE))"
con.sql(f"""CREATE TEMP TABLE v AS
SELECT e.user_id, e.item_id, it.category_id, it.brand_id, {D.format('e.occurred_at')} AS day
FROM event e JOIN item it USING (item_id)
WHERE e.event_type = 'view' AND {D.format('e.occurred_at')} BETWEEN {a} AND {b}""")

# %% Step 2 — 1 対多の集約｜会員 1 人の閲覧（多）を、カテゴリごとの件数（1 行）にまとめる。これが嗜好の特徴量になる。
one = con.sql("SELECT user_id FROM v GROUP BY 1 ORDER BY COUNT(*) DESC, user_id LIMIT 1 OFFSET 1000").fetchall()[0][0]
prof = con.sql(f"""
SELECT c.major || '／' || c.minor AS カテゴリ, COUNT(*) AS 閲覧数, ROUND(COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (), 3) AS 割合
FROM v JOIN category c USING (category_id) WHERE user_id = {one} GROUP BY 1 ORDER BY 閲覧数 DESC LIMIT 5
""").df()
show(prof, f"会員 {one} の閲覧のカテゴリ別の集約（上位 5）", index=False)

# %% Step 3 — ID の種類の多さ｜会員・商品・ブランド・カテゴリの種類数。多いほどワンホットでは扱いにくい。
card = con.sql("""SELECT COUNT(DISTINCT user_id) AS 会員, COUNT(DISTINCT item_id) AS 商品,
                  COUNT(DISTINCT brand_id) AS ブランド, COUNT(DISTINCT category_id) AS カテゴリ FROM v""").df()
show(card, "閲覧に現れた ID の種類数（開発期間）", index=False)

# %% Step 4 — 人気の偏り｜商品の閲覧数を多い順に並べ、上位 1%・10% の商品が全体の何割を占めるかを見る。
n = np.sort(con.sql("SELECT COUNT(*) FROM v GROUP BY item_id").df().iloc[:, 0].to_numpy())[::-1]
share = lambda q: n[: max(1, int(len(n) * q))].sum() / n.sum()
gini = 1 - 2 * np.cumsum(np.sort(n)).sum() / (len(n) * n.sum()) + 1 / len(n)
print(f"上位 1% の商品の閲覧の割合: {share(0.01):.3f}")
print(f"上位 10% の商品の閲覧の割合: {share(0.10):.3f}")
print(f"閲覧数のジニ係数: {gini:.3f}")
