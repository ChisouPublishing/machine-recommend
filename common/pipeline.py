"""第12章 12.3・12.4：多段の推薦の共通部品。12_pipeline/01_multistage.py と同じ評価のセッション・軽いランキング・評価の仕方を、
02（複数の候補源）と 03（DPP）から呼べるようにまとめたもの。中身は 01 の Step 1〜5 と同じ（01 の本文の説明を参照）。
"""
from __future__ import annotations

import sys

import implicit
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse

from .data import connect
from .paths import DATA, OUT, SEED
from .retrieval import interactions, true_top

sys.path.insert(0, str(DATA.parents[1]))
from truth import Truth  # noqa: E402  答え合わせ用

D = "date_diff('day', DATE '2025-04-07', CAST({} AS DATE))"


class Pipeline:
    def __init__(self):
        self.T = Truth(DATA)
        self.con = con = connect()
        self.cands = pd.read_pickle(OUT / "09_retrieval" / "cands_01.pkl")
        st = true_top()[["session_id", "user_id", "day"]]
        self.st = st[st.user_id.isin(self.cands["ALS（implicit）"])].reset_index(drop=True)
        self.it = con.sql("SELECT item_id, category_id - 1 AS c, brand_id - 1 AS b, gender, list_price FROM item ORDER BY item_id").df()
        self.IC, self.IB = self.it.c.to_numpy(), self.it.b.to_numpy()
        pp = con.sql(f"SELECT item_id - 1 AS i, {D.format('occurred_at')} AS d, COUNT(*) AS n FROM event WHERE event_type='purchase' GROUP BY ALL").df()
        daily = np.zeros((8000, 190)); daily[pp.i, pp.d] = pp.n
        self.pop7 = np.log1p(np.stack([daily[:, max(d - 7, 0):d].sum(1) for d in range(190)], 1))
        self.gender = con.sql('SELECT user_id, gender FROM "user"').df().set_index("user_id").gender
        stock = con.sql(f"SELECT item_id, {D.format('date')} AS d FROM item_availability").df()
        self.avail = set(zip(stock.item_id, stock.d))
        # ランキング：学習 day 105〜132 のトップの表示（5 分の 1）のクリックで、6 つの特徴量の軽い LightGBM（01 の (b2) と同じ）
        log = con.sql(f"""SELECT i.user_id, {D.format('i.shown_at')} AS day, i.item_id,
          EXISTS (SELECT 1 FROM event e WHERE e.impression_id = i.impression_id AND e.event_type = 'view')::INT AS y
          FROM impression i WHERE i.surface = 'top' AND {D.format('i.shown_at')} BETWEEN 105 AND 132 AND i.impression_id % 5 = 0
          ORDER BY i.impression_id""").df()
        self.rank_model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=SEED, verbose=-1,
                                             n_jobs=1, deterministic=True)
        self.rank_model.fit(np.column_stack([self.feats(log), self.als_score(self.als_until(104), log)]), log.y)
        self.als132 = self.als_until(132)

    def user_counts(self, keys):
        con = self.con
        con.register("k", keys)
        q = lambda col: con.sql(f"""SELECT k.user_id, k.day, it.{col} - 1 AS x, COUNT(*) AS n FROM k JOIN event e ON e.user_id = k.user_id
            AND e.event_type = 'view' AND {D.format('e.occurred_at')} BETWEEN k.day - 28 AND k.day - 1 JOIN item it USING (item_id) GROUP BY ALL""").df()
        pl = con.sql(f"""SELECT k.user_id, k.day, AVG(ln(l.unit_price)) AS lp FROM k JOIN "order" o ON o.user_id = k.user_id
            AND {D.format('o.ordered_at')} < k.day JOIN order_line l USING (order_id) GROUP BY ALL""").df()
        return q("category_id"), q("brand_id"), pl

    def feats(self, pairs):
        keys = pairs[["user_id", "day"]].drop_duplicates()
        cc, bc, pl = self.user_counts(keys)
        C = {(u, d): {} for u, d in zip(keys.user_id, keys.day)}
        for r in cc.itertuples():
            C[(r.user_id, r.day)][("c", r.x)] = r.n
        for r in bc.itertuples():
            C[(r.user_id, r.day)][("b", r.x)] = r.n
        lp = {(r.user_id, r.day): r.lp for r in pl.itertuples()}
        i = pairs.item_id.to_numpy() - 1
        it = self.it
        rows = [(C[(u, d)].get(("c", it.c[k]), 0), C[(u, d)].get(("b", it.b[k]), 0), lp.get((u, d), np.log(6500)))
                for u, d, k in zip(pairs.user_id, pairs.day, i)]
        a = np.array(rows, float)
        g = np.array([self.gender[u] for u in pairs.user_id], dtype=object)
        return np.column_stack([np.log1p(a[:, 0]), np.log1p(a[:, 1]), self.pop7[i, pairs.day.to_numpy()],
                                np.log(it.list_price.to_numpy()[i]) - a[:, 2],
                                ((it.gender.to_numpy()[i] == g) | (it.gender.to_numpy()[i] == "U")).astype(float)])

    @staticmethod
    def als_until(b):
        t = interactions(28, b)
        X = sparse.csr_matrix((t.w.astype(np.float32), (t.user_id, t.item_id - 1)), shape=(50_001, 8000))
        m = implicit.cpu.als.AlternatingLeastSquares(factors=64, regularization=0.05, iterations=15, random_state=SEED)
        m.fit(X, show_progress=False)
        return m

    @staticmethod
    def als_score(m, pairs):
        return np.einsum("ij,ij->i", m.user_factors[pairs.user_id.to_numpy()], m.item_factors[pairs.item_id.to_numpy() - 1])

    def score(self, cands: dict) -> pd.DataFrame:
        """会員ごとの候補（item_id の並び）を、評価のセッションごとに採点する。列：session_id・user_id・day・item_id・score・ok（在庫あり）。"""
        st = self.st
        pairs = pd.DataFrame([(s, u, d, i) for s, u, d in zip(st.session_id, st.user_id, st.day) for i in cands[u]],
                             columns=["session_id", "user_id", "day", "item_id"])
        X = np.column_stack([self.feats(pairs), self.als_score(self.als132, pairs)])
        pairs["score"] = self.rank_model.predict_proba(X)[:, 1]
        pairs["ok"] = [(i, d) in self.avail for i, d in zip(pairs.item_id, pairs.day)]
        return pairs

    def evaluate(self, sel: dict, sessions=None) -> dict:
        """6 枠の真のクリック確率の和（順位の効果込み）、在庫切れの数、異なるブランド・カテゴリの数。"""
        st = self.st if sessions is None else self.st[self.st.session_id.isin(sessions)]
        ss = st.session_id.to_numpy()
        items = np.array([sel[s] + [sel[s][-1]] * (6 - len(sel[s])) for s in ss])
        pc = self.T.p_click(ss, items, np.tile(np.arange(1, 7), (len(ss), 1)))
        return {"期待クリック数（6 枠）": pc.sum(1).mean(),
                "在庫切れ（件）": np.mean([sum((i, d) not in self.avail for i in sel[s]) for s, d in zip(ss, st.day)]),
                "ブランドの数": np.mean([len({self.IB[i - 1] for i in sel[s]}) for s in ss]),
                "カテゴリの数": np.mean([len({self.IC[i - 1] for i in sel[s]}) for s in ss])}
