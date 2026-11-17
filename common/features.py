"""第3章：トップのおすすめの表示ごとの特徴量（時点を守った集計）。第5章以降はこの表を共通に使う。

時点の約束：
  - 集計の特徴量（直近 7 日・28 日の閲覧数など）は、表示の日の 0 時より前の記録だけで作る（前日までの日次バッチ）。
  - 返品は、表示の日より前に返品された分だけを数える（注文日ではなく返品日で区切る）。
  - 閲覧の系列（直近 20 件）は、セッションの開始時刻より前の閲覧だけで作る。
  - 在庫は、表示の日より前の最新の週次スナップショットを使う。
"""
# %% Step 1 — 準備｜出力先と、日付を日番号（day 0 = 2025-04-07）に直す式を決める。
from __future__ import annotations

import time

import pandas as pd

from .data import connect, PERIODS
from .paths import DATA, OUT

FEATURES = OUT / "features" / f"top_{DATA.name}.parquet"         # データの版ごと（既定は default）
DAY = "date_diff('day', DATE '2025-04-07', CAST({} AS DATE))"

CAT_COLS = ["gender", "age_band", "region", "member_rank", "device", "body_size", "category_id", "major",
            "brand_id", "item_gender", "season", "price_tier", "position", "hour", "weekday"]
NUM_COLS = ["log_price", "price_gap", "days_since_reg", "days_since_release", "is_sale", "mail_opt_in",
            "u_views_7d", "u_views_28d", "u_purch_28d", "u_purch_all", "u_return_rate", "u_cat_views_28d",
            "u_brand_views_28d", "u_cat_purch_all", "i_views_7d", "i_purch_7d", "i_imp_28d", "i_click_28d",
            "c_purch_7d", "size_in_stock"]
SEQ_COLS = ["hist_items", "hist_cats"]
LABELS = ["clicked", "purchased"]


def build(con=None) -> pd.DataFrame:
    con = con or connect()
    # %% Step 2 — 表示とラベル｜トップの表示に、クリック・購入の有無（同じ impression_id の閲覧・購入）を付ける。
    con.sql(f"""
    CREATE OR REPLACE TEMP TABLE imp AS
    SELECT i.impression_id, i.user_id, i.session_id, i.item_id, i.position, i.shown_at,
           {DAY.format('i.shown_at')} AS day,
           EXISTS (SELECT 1 FROM event e WHERE e.impression_id = i.impression_id AND e.event_type = 'view') AS clicked,
           EXISTS (SELECT 1 FROM event e WHERE e.impression_id = i.impression_id AND e.event_type = 'purchase') AS purchased
    FROM impression i WHERE i.surface = 'top'
    """)
    # %% Step 3 — 日次の集計｜ユーザー×日、ユーザー×カテゴリ×日、商品×日の件数を作る（後で期間を区切って足す）。
    con.sql(f"""
    CREATE OR REPLACE TEMP TABLE ev AS
    SELECT e.user_id, e.item_id, it.category_id, it.brand_id, e.event_type, {DAY.format('e.occurred_at')} AS day
    FROM event e JOIN item it USING (item_id) WHERE e.event_type IN ('view', 'purchase');
    CREATE OR REPLACE TEMP TABLE u_day AS
    SELECT user_id, day, SUM((event_type = 'view')::INT) AS v, SUM((event_type = 'purchase')::INT) AS p FROM ev GROUP BY ALL;
    CREATE OR REPLACE TEMP TABLE uc_day AS
    SELECT user_id, category_id, day, SUM((event_type = 'view')::INT) AS v, SUM((event_type = 'purchase')::INT) AS p
    FROM ev GROUP BY ALL;
    CREATE OR REPLACE TEMP TABLE ub_day AS
    SELECT user_id, brand_id, day, COUNT(*) AS v FROM ev WHERE event_type = 'view' GROUP BY ALL;
    CREATE OR REPLACE TEMP TABLE i_day AS
    SELECT item_id, day, SUM((event_type = 'view')::INT) AS v, SUM((event_type = 'purchase')::INT) AS p FROM ev GROUP BY ALL;
    CREATE OR REPLACE TEMP TABLE c_day AS
    SELECT category_id, day, SUM((event_type = 'purchase')::INT) AS p FROM ev GROUP BY ALL;
    CREATE OR REPLACE TEMP TABLE i_imp_day AS
    SELECT item_id, day, COUNT(*) AS n, SUM(clicked::INT) AS c FROM imp GROUP BY ALL;
    """)
    # %% Step 4 — 時点を守った窓の集計｜表示の日 d に対し、d より前（d-7〜d-1、d-28〜d-1）の日だけを足す。
    con.sql("""
    CREATE OR REPLACE TEMP TABLE keys AS SELECT DISTINCT i.user_id, i.item_id, it.category_id, it.brand_id, i.day
    FROM imp i JOIN item it USING (item_id);
    CREATE OR REPLACE TEMP TABLE uf AS
    SELECT k.user_id, k.day,
           COALESCE(SUM(CASE WHEN u.day >= k.day - 7 THEN u.v END), 0) AS u_views_7d,
           COALESCE(SUM(CASE WHEN u.day >= k.day - 28 THEN u.v END), 0) AS u_views_28d,
           COALESCE(SUM(CASE WHEN u.day >= k.day - 28 THEN u.p END), 0) AS u_purch_28d
    FROM (SELECT DISTINCT user_id, day FROM keys) k
    LEFT JOIN u_day u ON u.user_id = k.user_id AND u.day < k.day AND u.day >= k.day - 28
    GROUP BY ALL;
    CREATE OR REPLACE TEMP TABLE ucf AS
    SELECT k.user_id, k.category_id, k.day,
           COALESCE(SUM(CASE WHEN u.day >= k.day - 28 THEN u.v END), 0) AS u_cat_views_28d,
           COALESCE(SUM(u.p), 0) AS u_cat_purch_all
    FROM (SELECT DISTINCT user_id, category_id, day FROM keys) k
    LEFT JOIN uc_day u ON u.user_id = k.user_id AND u.category_id = k.category_id AND u.day < k.day
    GROUP BY ALL;
    CREATE OR REPLACE TEMP TABLE ubf AS
    SELECT k.user_id, k.brand_id, k.day, COALESCE(SUM(u.v), 0) AS u_brand_views_28d
    FROM (SELECT DISTINCT user_id, brand_id, day FROM keys) k
    LEFT JOIN ub_day u ON u.user_id = k.user_id AND u.brand_id = k.brand_id AND u.day < k.day AND u.day >= k.day - 28
    GROUP BY ALL;
    CREATE OR REPLACE TEMP TABLE itf AS
    SELECT k.item_id, k.day,
           COALESCE(SUM(CASE WHEN x.day >= k.day - 7 THEN x.v END), 0) AS i_views_7d,
           COALESCE(SUM(CASE WHEN x.day >= k.day - 7 THEN x.p END), 0) AS i_purch_7d
    FROM (SELECT DISTINCT item_id, day FROM keys) k
    LEFT JOIN i_day x ON x.item_id = k.item_id AND x.day < k.day AND x.day >= k.day - 7
    GROUP BY ALL;
    CREATE OR REPLACE TEMP TABLE iif AS
    SELECT k.item_id, k.day, COALESCE(SUM(x.n), 0) AS i_imp_28d, COALESCE(SUM(x.c), 0) AS i_click_28d
    FROM (SELECT DISTINCT item_id, day FROM keys) k
    LEFT JOIN i_imp_day x ON x.item_id = k.item_id AND x.day < k.day AND x.day >= k.day - 28
    GROUP BY ALL;
    CREATE OR REPLACE TEMP TABLE cf AS
    SELECT k.category_id, k.day, COALESCE(SUM(x.p), 0) AS c_purch_7d
    FROM (SELECT DISTINCT category_id, day FROM keys) k
    LEFT JOIN c_day x ON x.category_id = k.category_id AND x.day < k.day AND x.day >= k.day - 7
    GROUP BY ALL;
    """)
    # %% Step 5 — 購買・返品の履歴｜全期間の購入数・平均購入価格と、表示の日より前に返品された割合を作る。
    con.sql(f"""
    CREATE OR REPLACE TEMP TABLE ol AS
    SELECT o.user_id, {DAY.format('o.ordered_at')} AS day, l.order_line_id, l.unit_price,
           {DAY.format('r.returned_at')} AS rday
    FROM order_line l JOIN "order" o USING (order_id) LEFT JOIN "return" r USING (order_line_id);
    CREATE OR REPLACE TEMP TABLE hf AS
    SELECT k.user_id, k.day, COUNT(ol.order_line_id) AS u_purch_all,
           COALESCE(SUM((ol.rday < k.day)::INT) * 1.0 / NULLIF(COUNT(ol.order_line_id), 0), 0) AS u_return_rate,
           ROUND(AVG(ln(ol.unit_price)), 10) AS u_log_price   -- 並列の足し算の順で末尾の桁が揺れるので丸める（付録 A）
    FROM (SELECT DISTINCT user_id, day FROM keys) k
    LEFT JOIN ol ON ol.user_id = k.user_id AND ol.day < k.day
    GROUP BY ALL;
    """)
    # %% Step 6 — 閲覧の系列｜セッションの開始より前の、直近 20 件の閲覧の商品とカテゴリを「^」区切りで並べる。
    con.sql("""
    CREATE OR REPLACE TEMP TABLE seq AS
    WITH s AS (SELECT DISTINCT session_id, user_id FROM imp),
    v AS (
      SELECT s.session_id, e.item_id, it.category_id, e.occurred_at,
             ROW_NUMBER() OVER (PARTITION BY s.session_id ORDER BY e.occurred_at DESC, e.event_id DESC) AS rn
      FROM s JOIN session ss USING (session_id)
      JOIN event e ON e.user_id = s.user_id AND e.event_type = 'view' AND e.occurred_at < ss.started_at
      JOIN item it ON it.item_id = e.item_id)
    SELECT session_id,
           string_agg(item_id::VARCHAR, '^' ORDER BY rn DESC) AS hist_items,
           string_agg(category_id::VARCHAR, '^' ORDER BY rn DESC) AS hist_cats
    FROM v WHERE rn <= 20 GROUP BY session_id
    """)
    # %% Step 7 — 在庫｜表示の日より前の最新の週次スナップショットで、会員のサイズの在庫があるかを見る。
    con.sql(f"""
    CREATE OR REPLACE TEMP TABLE stock AS
    SELECT k.user_id, k.item_id, k.day,
           COALESCE(MAX(CASE WHEN inv.stock > 0 AND (s.size = u.body_size OR s.size = u.shoe_size OR s.size = 'F')
                    THEN 1 ELSE 0 END), 0) AS size_in_stock
    FROM (SELECT DISTINCT user_id, item_id, day FROM keys) k
    JOIN "user" u USING (user_id) JOIN sku s ON s.item_id = k.item_id
    LEFT JOIN inventory inv ON inv.sku_id = s.sku_id
         AND {DAY.format('inv.snapshot_date')} = (k.day - 1) - ((k.day - 1 + 1) % 7)
    GROUP BY ALL
    """)
    # %% Step 8 — 結合｜表示に会員・商品・文脈・集計・系列を付け、学習用の 1 枚の表にする。
    df = con.sql(f"""
    SELECT i.impression_id, i.user_id, i.item_id, i.session_id, i.day, i.shown_at,
           u.gender, u.age_band, u.region, u.member_rank, u.device, u.body_size, u.mail_opt_in::INT AS mail_opt_in,
           i.day - {DAY.format('u.registered_at')} AS days_since_reg,
           it.category_id, c.major, it.brand_id, it.gender AS item_gender, it.season, b.price_tier,
           ln(it.list_price) AS log_price, ln(it.list_price) - COALESCE(hf.u_log_price, ln(6500)) AS price_gap,
           i.day - {DAY.format('it.release_date')} AS days_since_release,
           (i.day BETWEEN 81 AND 97 AND it.season = 'SS')::INT AS is_sale,
           i.position, hour(i.shown_at) AS hour, dayofweek(i.shown_at) AS weekday,
           uf.u_views_7d, uf.u_views_28d, uf.u_purch_28d, hf.u_purch_all, hf.u_return_rate,
           ucf.u_cat_views_28d, ucf.u_cat_purch_all, ubf.u_brand_views_28d,
           itf.i_views_7d, itf.i_purch_7d, iif.i_imp_28d, iif.i_click_28d, cf.c_purch_7d,
           st.size_in_stock, COALESCE(seq.hist_items, '') AS hist_items, COALESCE(seq.hist_cats, '') AS hist_cats,
           i.clicked::INT AS clicked, i.purchased::INT AS purchased
    FROM imp i
    JOIN "user" u USING (user_id) JOIN item it USING (item_id) JOIN category c USING (category_id)
    JOIN brand b ON b.brand_id = it.brand_id
    JOIN uf ON uf.user_id = i.user_id AND uf.day = i.day
    JOIN hf ON hf.user_id = i.user_id AND hf.day = i.day
    JOIN ucf ON ucf.user_id = i.user_id AND ucf.category_id = it.category_id AND ucf.day = i.day
    JOIN ubf ON ubf.user_id = i.user_id AND ubf.brand_id = it.brand_id AND ubf.day = i.day
    JOIN itf ON itf.item_id = i.item_id AND itf.day = i.day
    JOIN iif ON iif.item_id = i.item_id AND iif.day = i.day
    JOIN cf ON cf.category_id = it.category_id AND cf.day = i.day
    JOIN stock st ON st.user_id = i.user_id AND st.item_id = i.item_id AND st.day = i.day
    LEFT JOIN seq ON seq.session_id = i.session_id
    WHERE i.day >= {PERIODS['dev'][0]}
    ORDER BY i.impression_id
    """).df()
    return df


def load_top(rebuild: bool = False) -> pd.DataFrame:
    """特徴量の表を読む（無ければ作って保存する）。"""
    if rebuild or not FEATURES.exists():
        t = time.time()
        df = build()
        FEATURES.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(FEATURES, index=False)
        print(f"[time] 特徴量の作成 {time.time() - t:.1f} 秒")
    return pd.read_parquet(FEATURES)


def split(df: pd.DataFrame, a: int, b: int) -> pd.DataFrame:
    """day が a 以上 b 以下の行。"""
    return df[(df.day >= a) & (df.day <= b)]
