"""第12章：多段の推薦。候補生成（ALS の 100 件）→ ランキング（軽い LightGBM）→ 再ランキング（在庫・ブランドの偏り・多様性）で 6 枠を出し、
真のクリック確率で成果を、段ごとの時間でレイテンシを測る。"""
# %% Step 1 — 準備｜評価の期間（day 133〜146）の状態を記録したセッション。候補は第9章の ALS（学習 day 28〜132）の上位 100 件。
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import DATA, SEED, connect, show
from common.retrieval import true_top

sys.path.insert(0, str(DATA.parents[1]))
from truth import Truth  # noqa: E402  答え合わせ用

T = Truth(DATA)
con = connect()
D = "date_diff('day', DATE '2025-04-07', CAST({} AS DATE))"
cands = pd.read_pickle("outputs/09_retrieval/cands_01.pkl")["ALS（implicit）"]
st = true_top()[["session_id", "user_id", "day"]]
st = st[st.user_id.isin(cands)].reset_index(drop=True)
it = con.sql("SELECT item_id, category_id - 1 AS c, brand_id - 1 AS b, gender, list_price FROM item ORDER BY item_id").df()
print(f"セッション {len(st):,}、候補 100 件／人")

# %% Step 2 — 軽いランキングの特徴量｜どの（会員, 商品）にも計算できる 5 つ：カテゴリ・ブランドの閲覧数（前日までの 28 日）、人気（7 日）、
#               価格の差（会員の過去の平均購入価格との対数の差）、性別が合うか。
def user_counts(keys):
    con.register("k", keys)
    q = lambda col: con.sql(f"""SELECT k.user_id, k.day, it.{col} - 1 AS x, COUNT(*) AS n FROM k JOIN event e ON e.user_id = k.user_id
        AND e.event_type = 'view' AND {D.format('e.occurred_at')} BETWEEN k.day - 28 AND k.day - 1 JOIN item it USING (item_id) GROUP BY ALL""").df()
    pl = con.sql(f"""SELECT k.user_id, k.day, AVG(ln(l.unit_price)) AS lp FROM k JOIN "order" o ON o.user_id = k.user_id
        AND {D.format('o.ordered_at')} < k.day JOIN order_line l USING (order_id) GROUP BY ALL""").df()
    return q("category_id"), q("brand_id"), pl


pp = con.sql(f"SELECT item_id - 1 AS i, {D.format('occurred_at')} AS d, COUNT(*) AS n FROM event WHERE event_type='purchase' GROUP BY ALL").df()
daily = np.zeros((8000, 190)); daily[pp.i, pp.d] = pp.n
pop7 = np.log1p(np.stack([daily[:, max(d - 7, 0):d].sum(1) for d in range(190)], 1))
gender = con.sql('SELECT user_id, gender FROM "user"').df().set_index("user_id").gender


def feats(pairs):
    """pairs：user_id, day, item_id の表。5 つの特徴量の行列を返す。"""
    keys = pairs[["user_id", "day"]].drop_duplicates()
    cc, bc, pl = user_counts(keys)
    C = {(u, d): {} for u, d in zip(keys.user_id, keys.day)}
    for r in cc.itertuples():
        C[(r.user_id, r.day)][("c", r.x)] = r.n
    for r in bc.itertuples():
        C[(r.user_id, r.day)][("b", r.x)] = r.n
    lp = {(r.user_id, r.day): r.lp for r in pl.itertuples()}
    i = pairs.item_id.to_numpy() - 1
    rows = [(C[(u, d)].get(("c", it.c[k]), 0), C[(u, d)].get(("b", it.b[k]), 0), lp.get((u, d), np.log(6500)))
            for u, d, k in zip(pairs.user_id, pairs.day, i)]
    a = np.array(rows, float)
    g = np.array([gender[u] for u in pairs.user_id], dtype=object)
    return np.column_stack([np.log1p(a[:, 0]), np.log1p(a[:, 1]), pop7[i, pairs.day.to_numpy()],
                            np.log(it.list_price.to_numpy()[i]) - a[:, 2],
                            ((it.gender.to_numpy()[i] == g) | (it.gender.to_numpy()[i] == "U")).astype(float)])


# %% Step 3 — ランキングのモデル｜学習 day 105〜132 のトップの表示（1〜6 位）のクリックで、軽い LightGBM を学ぶ（順位は特徴量に入れない）。
#               2 つ作る：5 つの特徴量だけのものと、候補生成の点数（ALS の内積）を 6 つ目に足したもの。ALS の点数は時点を守り、
#               学習の行には day 104 までで学んだ ALS、評価の候補には day 132 までで学んだ ALS（候補を作ったもの）を使う。
import implicit
from scipy import sparse

from common.retrieval import interactions


def als_until(b):
    t = interactions(28, b)
    X = sparse.csr_matrix((t.w.astype(np.float32), (t.user_id, t.item_id - 1)), shape=(50_001, 8000))
    m = implicit.cpu.als.AlternatingLeastSquares(factors=64, regularization=0.05, iterations=15, random_state=SEED)
    m.fit(X, show_progress=False)
    return m


def als_score(m, pairs):
    return np.einsum("ij,ij->i", m.user_factors[pairs.user_id.to_numpy()], m.item_factors[pairs.item_id.to_numpy() - 1])


log = con.sql(f"""SELECT i.user_id, {D.format('i.shown_at')} AS day, i.item_id,
  EXISTS (SELECT 1 FROM event e WHERE e.impression_id = i.impression_id AND e.event_type = 'view')::INT AS y
  FROM impression i WHERE i.surface = 'top' AND {D.format('i.shown_at')} BETWEEN 105 AND 132 AND i.impression_id % 5 = 0
  ORDER BY i.impression_id""").df()
Flog = feats(log)
mk = lambda: lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=SEED, verbose=-1,
                                n_jobs=1, deterministic=True)
rank_model = mk().fit(Flog, log.y)
rank_model2 = mk().fit(np.column_stack([Flog, als_score(als_until(104), log)]), log.y)
print(f"ランキングの学習 {len(log):,} 行（クリック率 {log.y.mean():.4f}）")

# %% Step 4 — 5 つの出し方｜(a) 候補の上位 6、(b1)(b2) ランキングの上位 6、(c) (b2)＋在庫切れを除く、(d) (c)＋同じブランドは 2 件まで・MMR。
stock = con.sql(f"SELECT item_id, {D.format('date')} AS d FROM item_availability").df()
avail = set(zip(stock.item_id, stock.d))
pairs = pd.DataFrame([(s, u, d, i) for s, u, d in zip(st.session_id, st.user_id, st.day) for i in cands[u]],
                     columns=["session_id", "user_id", "day", "item_id"])
X = feats(pairs)
pairs["score1"] = rank_model.predict_proba(X)[:, 1]
X2 = np.column_stack([X, als_score(als_until(132), pairs)])
t0 = time.perf_counter()
pairs["score"] = rank_model2.predict_proba(X2)[:, 1]
rank_ms = (time.perf_counter() - t0) / len(st) * 1000
pairs["ok"] = [(i, d) in avail for i, d in zip(pairs.item_id, pairs.day)]


IC, IB = it.c.to_numpy(), it.b.to_numpy()


def mmr(g, lam=0.7, brand_cap=2):
    """点数と、すでに選んだ商品とのカテゴリ・ブランドの重なりの罰を両方見て、1 件ずつ選ぶ（同じブランドは brand_cap 件まで）。"""
    g = g[g.ok]
    ids = g.item_id.to_numpy()
    sc = g.score.to_numpy() / g.score.max()
    c, b = IC[ids - 1], IB[ids - 1]
    sim = np.zeros(len(ids))                                  # すでに選んだ商品との似ている度合いの最大
    left = np.ones(len(ids), bool)
    chosen, cnt = [], {}
    while len(chosen) < 6 and left.any():
        val = np.where(left, lam * sc - (1 - lam) * sim, -np.inf)
        k = int(np.argmax(val))
        left[k] = False
        if cnt.get(b[k], 0) >= brand_cap:
            continue
        chosen.append(ids[k]); cnt[b[k]] = cnt.get(b[k], 0) + 1
        sim = np.maximum(sim, 0.5 * (c == c[k]) + 0.5 * (b == b[k]))
    return chosen


t0 = time.perf_counter()
out = {"(a) 候補の上位 6": {}, "(b1) ランキング（5 つの特徴量）": {}, "(b2) ランキング（＋候補生成の点数）": {},
       "(c) ＋在庫切れを除く": {}, "(d) ＋ブランド 2 件まで・MMR": {}}
for s, g in pairs.groupby("session_id", sort=False):
    out["(a) 候補の上位 6"][s] = list(g.item_id[:6])
    out["(b1) ランキング（5 つの特徴量）"][s] = list(g.sort_values("score1", ascending=False).item_id[:6])
    gs = g.sort_values("score", ascending=False)
    out["(b2) ランキング（＋候補生成の点数）"][s] = list(gs.item_id[:6])
    out["(c) ＋在庫切れを除く"][s] = list(gs[gs.ok].item_id[:6])
mmr_t0 = time.perf_counter()
for s, g in pairs.groupby("session_id", sort=False):
    out["(d) ＋ブランド 2 件まで・MMR"][s] = mmr(g)
rerank_ms = (time.perf_counter() - mmr_t0) / len(st) * 1000

# %% Step 5 — 評価｜6 枠の真のクリック確率の和（順位の効果込み）、在庫切れの数、異なるブランド・カテゴリの数。
rows = []
for name, sel in out.items():
    ss = st.session_id.to_numpy()
    items = np.array([sel[s] + [sel[s][-1]] * (6 - len(sel[s])) for s in ss])
    pc = T.p_click(ss, items, np.tile(np.arange(1, 7), (len(ss), 1)))
    nos = np.mean([sum((i, d) not in avail for i in sel[s]) for s, d in zip(ss, st.day)])
    rows.append((name, pc.sum(1).mean(), nos, np.mean([len({it.b[i - 1] for i in sel[s]}) for s in ss]),
                 np.mean([len({it.c[i - 1] for i in sel[s]}) for s in ss])))
show(pd.DataFrame(rows, columns=["出し方", "期待クリック数（6 枠）", "在庫切れ（件）", "ブランドの数", "カテゴリの数"]),
     f"多段の推薦の比較（{len(st):,} セッション、評価 day 133〜146、合成データ）", index=False, digits=4)
print(f"[time] ランキング（100 件の採点）1 セッションあたり {rank_ms:.2f} ms・再ランキング（MMR）{rerank_ms:.2f} ms")
