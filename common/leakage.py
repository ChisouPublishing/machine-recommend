"""第3章：特徴量の時点の検査。集計の特徴量を「表示の日より前の記録だけ」で数え直し、一致するかを確かめる。"""
from __future__ import annotations

import pandas as pd

D = "date_diff('day', DATE '2025-04-07', CAST({} AS DATE))"


def audit_user_views(df: pd.DataFrame, con, col: str = "u_views_28d", window: int = 28,
                     n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """df の col（会員の直近 window 日の閲覧数）を、表示の日 d の前日までの閲覧で数え直す。食い違う行を返す。"""
    s = df.sample(min(n, len(df)), random_state=seed)[["impression_id", "user_id", "day", col]]
    con.register("s", s)
    re = con.sql(f"""
    SELECT s.impression_id, COUNT(e.event_id) AS recount
    FROM s LEFT JOIN event e ON e.user_id = s.user_id AND e.event_type = 'view'
         AND {D.format('e.occurred_at')} < s.day AND {D.format('e.occurred_at')} >= s.day - {window}
    GROUP BY 1""").df()
    con.unregister("s")
    m = s.merge(re, on="impression_id")
    return m[m[col] != m.recount]


def audit_sequence(df: pd.DataFrame, con, n: int = 500, seed: int = 0) -> pd.DataFrame:
    """閲覧の系列（hist_items）が、セッションの開始より前の閲覧だけでできているかを数え直す。"""
    s = df.drop_duplicates("session_id").sample(min(n, df.session_id.nunique()), random_state=seed)
    s = s[["session_id", "user_id", "hist_items"]]
    con.register("s", s)
    re = con.sql("""
    WITH v AS (SELECT s.session_id, e.item_id,
                      ROW_NUMBER() OVER (PARTITION BY s.session_id ORDER BY e.occurred_at DESC, e.event_id DESC) AS rn
               FROM s JOIN session ss USING (session_id)
               JOIN event e ON e.user_id = s.user_id AND e.event_type = 'view' AND e.occurred_at < ss.started_at)
    SELECT session_id, string_agg(item_id::VARCHAR, '^' ORDER BY rn DESC) AS recount FROM v WHERE rn <= 20 GROUP BY 1""").df()
    con.unregister("s")
    m = s.merge(re, on="session_id", how="left").fillna({"recount": ""})
    return m[m.hist_items != m.recount]
