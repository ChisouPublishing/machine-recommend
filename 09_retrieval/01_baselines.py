"""第9章：候補生成の強いベースライン（人気・共起）と、行列分解（implicit の ALS・BPR）を同じ評価で比べる。"""
# %% Step 1 — データ｜学習 day 28〜132 の会員×商品の行動、評価 day 133〜146 の正解（5,000 人）、真の上位 20 件（約 1,500 セッション）。
import time

import implicit
import numpy as np
import pandas as pd
from scipy import sparse

from common import SEED, connect, show
from common.retrieval import TRAIN, eval_users, interactions, recall, true_recall, true_top

tr = interactions(*TRAIN)
users, truth = eval_users()
tt = true_top()
need = np.union1d(users, tt.user_id.unique())                 # 候補を作る会員
n_u, n_i = int(tr.user_id.max()) + 1, 8000
X = sparse.csr_matrix((tr.w.astype(np.float32), (tr.user_id, tr.item_id - 1)), shape=(n_u, n_i))   # 列 = item_id − 1
gender = connect().sql('SELECT user_id, gender FROM "user"').df().set_index("user_id").gender
ig = connect().sql("SELECT item_id, gender FROM item").df().set_index("item_id").gender
K, rows, cands_all = 100, [], {}


def evaluate(name, cands, secs):
    print(f"[time] {name} {secs:.1f} 秒")
    cands_all[name] = cands
    rows.append({"方法": name, **recall({u: cands.get(u, []) for u in users}, truth), **true_recall(cands, tt)})


# %% Step 2 — 人気｜学習期間の最後の 7 日の行動の多い順（会員の性別に合う商品だけ）。全員に同じ候補を出す。
t = time.time()
last = tr[tr.last_day >= TRAIN[1] - 6].groupby("item_id").w.sum().sort_values(ascending=False)
top = {g: [i for i in last.index if ig[i] in (g, "U")][:K] for g in ("M", "F")}
evaluate("人気（直近 7 日）", {u: top[gender[u]] for u in need}, time.time() - t)

# %% Step 3 — 共起｜同じ会員が行動した商品どうしのコサイン類似度。会員の直近 20 件の商品と似た商品の点数を足して並べる。
t = time.time()
B = (X > 0).astype(np.float32)
C = (B.T @ B).toarray()
np.fill_diagonal(C, 0)
deg = np.sqrt(np.asarray(B.sum(0)).ravel()) + 1e-9
S = C / deg[:, None] / deg[None, :]
recent = tr.sort_values("last_day").groupby("user_id").tail(20)
R = sparse.csr_matrix((np.ones(len(recent), np.float32), (recent.user_id, recent.item_id - 1)), shape=(n_u, n_i))
sc = R[need] @ S
cands = {u: list(np.argsort(-sc[k])[:K] + 1) for k, u in enumerate(need)}
evaluate("共起（アイテム間の類似）", cands, time.time() - t)

# %% Step 4 — 発展版：implicit の ALS と BPR｜会員×商品の行列を、潜在因子の内積に分解する（行動の重みを確信度として使う）。
for name, M in (("ALS（implicit）", implicit.cpu.als.AlternatingLeastSquares(factors=64, regularization=0.05, iterations=15,
                                                                            random_state=SEED)),
                ("BPR（implicit）", implicit.cpu.bpr.BayesianPersonalizedRanking(factors=64, iterations=100, random_state=SEED))):
    t = time.time()
    M.fit(X, show_progress=False)
    ids, _ = M.recommend(need, X[need], N=K, filter_already_liked_items=False)
    evaluate(name, {u: list(r + 1) for u, r in zip(need, ids)}, time.time() - t)

# %% Step 5 — 比較｜Recall@K（評価の期間の実際の行動）と、真の上位 20 件の再現率（候補 100 件）。
show(pd.DataFrame(rows).set_index("方法"), f"候補生成の比較（評価 day 133〜146、{len(users):,} 人・真の上位は {len(tt):,} セッション、合成データ）")
pd.to_pickle(cands_all, "outputs/09_retrieval/cands_01.pkl")
