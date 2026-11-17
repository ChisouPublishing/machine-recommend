"""第13章：特徴量ストアの最小実装（DuckDB＋Parquet）。日次の集計を「いつの時点の値か」付きで保存し、学習用（時点指定）と推論用（最新）を同じ定義で出す。"""
# %% Step 1 — 日次の集計を保存｜会員ごとの直近 7 日・28 日の閲覧数を、日ごとに計算して Parquet に書く（as_of = その日の 0 時の値）。
import duckdb
import numpy as np
import pandas as pd

from common import OUT, connect, show
from common.features import load_top, split

store = OUT / "13_ops" / "store"
store.mkdir(parents=True, exist_ok=True)
con = connect()
D = "date_diff('day', DATE '2025-04-07', CAST({} AS DATE))"
con.sql(f"""
COPY (
  WITH d AS (SELECT user_id, {D.format('occurred_at')} AS day, COUNT(*) AS v FROM event WHERE event_type = 'view' GROUP BY ALL),
  days AS (SELECT range AS as_of FROM range(28, 182)),
  u AS (SELECT DISTINCT user_id FROM d)
  SELECT u.user_id, days.as_of,
         COALESCE(SUM(CASE WHEN d.day >= days.as_of - 7 THEN d.v END), 0) AS u_views_7d,
         COALESCE(SUM(d.v), 0) AS u_views_28d
  FROM u CROSS JOIN days LEFT JOIN d ON d.user_id = u.user_id AND d.day < days.as_of AND d.day >= days.as_of - 28
  GROUP BY ALL HAVING COALESCE(SUM(d.v), 0) > 0
) TO '{store}/user_views.parquet' (FORMAT PARQUET)""")
fs = duckdb.connect()
fs.sql(f"CREATE VIEW user_views AS SELECT * FROM read_parquet('{store}/user_views.parquet')")
print("保存した行数:", f"{fs.sql('SELECT COUNT(*) FROM user_views').fetchall()[0][0]:,}")


# %% Step 2 — 学習用｜（会員, 日）の一覧に、その日の値を結合する（時点指定の結合。as_of がその日と同じ行だけを使う）。
def historical(keys: pd.DataFrame) -> pd.DataFrame:
    fs.register("k", keys)
    return fs.sql("""SELECT k.*, COALESCE(v.u_views_7d, 0) AS u_views_7d, COALESCE(v.u_views_28d, 0) AS u_views_28d
                     FROM k LEFT JOIN user_views v ON v.user_id = k.user_id AND v.as_of = k.day ORDER BY k.user_id, k.day""").df()


# %% Step 3 — 推論用｜指定した日の最新の値を返す（本番の「いま」を、日 d として渡す）。
def online(user_ids, day: int) -> pd.DataFrame:
    fs.register("u", pd.DataFrame({"user_id": user_ids}))
    return fs.sql(f"""SELECT u.user_id, COALESCE(v.u_views_7d, 0) AS u_views_7d, COALESCE(v.u_views_28d, 0) AS u_views_28d
                      FROM u LEFT JOIN user_views v ON v.user_id = u.user_id AND v.as_of = {day} ORDER BY u.user_id""").df()


# %% Step 4 — 一貫性の確認｜(1) 学習用の値が、第3章の特徴量の表の値と一致するか。(2) 日 140 の推論用の値が、学習用で日 140 を指定した値と一致するか。
f3 = split(load_top(), 133, 146)[["user_id", "day", "u_views_7d", "u_views_28d"]].drop_duplicates(["user_id", "day"])
h = historical(f3[["user_id", "day"]])
m = f3.merge(h, on=["user_id", "day"], suffixes=("_第3章", "_ストア"))
bad = int(((m.u_views_7d_第3章 != m.u_views_7d_ストア) | (m.u_views_28d_第3章 != m.u_views_28d_ストア)).sum())
print(f"(1) 第3章の表との食い違い: {bad} 件 / {len(m):,} 件")
users = np.sort(f3[f3.day == 140].user_id.unique())[:1000]
on = online(users, 140)
hi = historical(pd.DataFrame({"user_id": users, "day": 140}))
print(f"(2) 推論用と学習用の食い違い: {int((on.u_views_28d.to_numpy() != hi.u_views_28d.to_numpy()).sum())} 件 / {len(users):,} 人")
show(on.head(3).astype(int), "推論用の値（日 140、先頭 3 人）", index=False)
