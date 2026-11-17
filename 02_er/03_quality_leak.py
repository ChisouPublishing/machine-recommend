"""第2章：ER の構造から、データ品質の問題とリークの源泉を点検する。"""
# %% Step 1 — 準備｜共通処理の接続と、日付を日番号にする式。
from common import connect, show

con = connect()
D = "date_diff('day', DATE '2025-04-07', CAST({} AS DATE))"

# %% Step 2 — 時刻の整合｜子の記録が親より前に起きていないか（表示より前のクリック、注文より前の返品）。
q = f"""
SELECT 'クリックが表示より前' AS 点検, COUNT(*) AS 件数 FROM event e JOIN impression i USING (impression_id)
 WHERE e.occurred_at < i.shown_at
UNION ALL SELECT '返品が注文より前', COUNT(*) FROM "return" r JOIN order_line l USING (order_line_id)
 JOIN "order" o USING (order_id) WHERE r.returned_at < o.ordered_at
UNION ALL SELECT '発売前に表示', COUNT(*) FROM impression i JOIN item it USING (item_id)
 WHERE CAST(i.shown_at AS DATE) < it.release_date
UNION ALL SELECT '登録前に来訪', COUNT(*) FROM session s JOIN "user" u USING (user_id)
 WHERE CAST(s.started_at AS DATE) < u.registered_at
"""
show(con.sql(q).df(), "時刻の整合（0 件が正しい）", index=False)

# %% Step 3 — リークの源泉 1：後から確定する記録｜返品は注文の 3〜14 日後に確定する。注文日で数えると未来を見る。
q = f"""
WITH x AS (SELECT {D.format('o.ordered_at')} AS od, {D.format('r.returned_at')} AS rd
           FROM order_line l JOIN "order" o USING (order_id) JOIN "return" r USING (order_line_id))
SELECT COUNT(*) AS 返品数, AVG(rd - od) AS 確定までの平均日数, MAX(rd - od) AS 最大日数,
       AVG((rd > 146 AND od <= 146)::INT) AS 開発期間の末に未確定の割合
FROM x
"""
show(con.sql(q).df(), "返品の確定の遅れ", index=False, digits=3)

# %% Step 4 — リークの源泉 2：週次のスナップショット｜在庫を表示の日の「後」のスナップショットで結合すると、未来の在庫を使う。
q = f"""
WITH imp AS (SELECT i.impression_id, i.item_id, {D.format('i.shown_at')} AS d FROM impression i
             WHERE i.surface = 'top' AND {D.format('i.shown_at')} BETWEEN 28 AND 146),
inv AS (SELECT s.item_id, {D.format('v.snapshot_date')} AS sd, MAX((v.stock > 0)::INT) AS ok
        FROM inventory v JOIN sku s USING (sku_id) GROUP BY ALL)
SELECT AVG(p.ok) AS 前のスナップショットで在庫あり, AVG(n.ok) AS 後のスナップショットで在庫あり,
       AVG((p.ok <> n.ok)::INT) AS 食い違う割合
FROM imp
JOIN inv p ON p.item_id = imp.item_id AND p.sd = (imp.d - 1) - (imp.d % 7)
JOIN inv n ON n.item_id = imp.item_id AND n.sd = (imp.d - 1) - (imp.d % 7) + 7
"""
show(con.sql(q).df(), "在庫の結合の時点（トップの表示、開発期間）", index=False, digits=3)

# %% Step 5 — リークの源泉 3：ラベルと同じ記録｜表示の後のクリックを集計に含めると、ラベルそのものが特徴量に入る（表示の 7 件に 1 件で数える）。
q = f"""
WITH s AS (SELECT i.impression_id, i.item_id, i.user_id, i.shown_at FROM impression i
           WHERE i.surface = 'top' AND {D.format('i.shown_at')} BETWEEN 140 AND 146 AND i.impression_id % 7 = 0),
lab AS (SELECT DISTINCT impression_id FROM event WHERE event_type = 'view' AND impression_id IS NOT NULL)
SELECT (l.impression_id IS NOT NULL)::INT AS クリック,
       AVG((SELECT COUNT(*) FROM event e WHERE e.user_id = s.user_id AND e.item_id = s.item_id
            AND e.event_type = 'view' AND CAST(e.occurred_at AS DATE) = CAST(s.shown_at AS DATE))) AS 同じ日の同じ商品の閲覧数
FROM s LEFT JOIN lab l USING (impression_id) GROUP BY 1 ORDER BY 1
"""
show(con.sql(q).df(), "「同じ日」の閲覧を数えた特徴量（クリックの有無別の平均）", index=False, digits=3)
