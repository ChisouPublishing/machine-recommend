"""第9章：候補生成の共通処理（学習用の行動の表、評価の正解、Recall@K、真の上位 K との一致）。

学習：開発期間の day 28〜132 の行動（閲覧・カート・購入。外部からの直帰の閲覧は除く）。
評価：day 133〜146 にその会員が閲覧・カート・購入した商品を正解とし、会員ごとに K 件の候補に何割入るか（Recall@K）。
答え合わせ：状態を記録したセッション（付録 B）では、表示できる全商品の真のクリック確率が分かる。真の上位 20 件のうち、
          候補の K 件に何件入るかも測る（「本当に良い商品を候補に残せたか」）。
"""
# %% Step 1 — 学習と評価の行動｜会員×商品の行動の数（閲覧 1・カート 3・購入 5 の重み）と、評価の正解の集合。
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common.data import connect
from common.paths import DATA, SEED

D = "date_diff('day', DATE '2025-04-07', CAST({} AS DATE))"
TRAIN, EVAL = (28, 132), (133, 146)


def interactions(a: int, b: int) -> pd.DataFrame:
    con = connect()
    return con.sql(f"""
    SELECT user_id, item_id, SUM(CASE event_type WHEN 'view' THEN 1 WHEN 'cart' THEN 3 ELSE 5 END) AS w,
           MAX({D.format('occurred_at')}) AS last_day
    FROM event WHERE event_type IN ('view', 'cart', 'purchase') AND source <> 'landing'
      AND {D.format('occurred_at')} BETWEEN {a} AND {b}
    GROUP BY 1, 2 ORDER BY 1, 2""").df()                     # 行の順を固定する（学習の順が変わると結果が変わるため）


def eval_users(n: int = 5000, seed: int = SEED) -> tuple[np.ndarray, dict]:
    """評価の期間に行動のあった会員から n 人を選び、正解の商品の集合を返す。"""
    ev = interactions(*EVAL)
    users = np.sort(ev.user_id.unique())
    pick = np.sort(np.random.default_rng(seed).choice(users, min(n, len(users)), replace=False))
    ev = ev[ev.user_id.isin(pick)]
    return pick, ev.groupby("user_id").item_id.apply(set).to_dict()


def recall(cands: dict, truth: dict, ks=(20, 50, 100)) -> dict:
    out = {}
    for k in ks:
        out[f"Recall@{k}"] = float(np.mean([len(set(cands.get(u, [])[:k]) & t) / len(t) for u, t in truth.items()]))
    return out


# %% Step 2 — 真の上位 K｜評価の期間の状態を記録したセッションで、その日に表示できた全商品の真のクリック確率の上位 20 件を作る。
def true_top(n_top: int = 20, max_sessions: int = 1500) -> pd.DataFrame:
    sys.path.insert(0, str(DATA.parents[1]))
    from truth import Truth
    T = Truth(DATA)
    con = connect()
    st = T.state.reset_index()
    st = st[(st.day >= EVAL[0]) & (st.day <= EVAL[1])].sort_values("session_id").head(max_sessions)
    gender = con.sql("SELECT user_id, gender FROM \"user\"").df().set_index("user_id").gender
    av = con.sql(f"SELECT item_id, {D.format('date')} AS d FROM item_availability").df()
    ig = con.sql("SELECT item_id, gender FROM item ORDER BY item_id").df().set_index("item_id").gender
    rows = []
    for (d,), g in st.groupby(["day"]):
        pool_all = av[av.d == d].item_id.to_numpy()
        for gen, gg in g.groupby(g.user_id.map(gender)):
            pool = pool_all[np.isin(ig.loc[pool_all].to_numpy(), [gen, "U"])]
            pc = T.p_click(gg.session_id.to_numpy(), np.broadcast_to(pool, (len(gg), len(pool))), np.ones((len(gg), len(pool))))
            top = pool[np.argsort(-pc, 1)[:, :n_top]]
            rows += [(s, u, int(d), list(t)) for s, u, t in zip(gg.session_id, gg.user_id, top)]
    return pd.DataFrame(rows, columns=["session_id", "user_id", "day", "true_top"])


def true_recall(cands: dict, tt: pd.DataFrame, ks=(100,)) -> dict:
    """真の上位 20 件のうち、会員の候補の上位 K 件に入った割合の平均（会員の候補は評価の期間の始めに 1 回作ったもの）。"""
    out = {}
    for k in ks:
        v = [len(set(cands.get(u, [])[:k]) & set(t)) / len(t) for u, t in zip(tt.user_id, tt.true_top) if u in cands]
        out[f"真の上位20の再現率@{k}"] = float(np.mean(v))
    return out
