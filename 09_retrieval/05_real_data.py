"""第9章：実データ（UCI Online Retail II、CC BY 4.0）で、候補生成の順位が合成データと同じになるかを確かめる。

出典：Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D
"""
# %% Step 1 — データ｜取消（Invoice が C で始まる）・数量 0 以下・会員番号の無い行を除き、2011-09-30 までを学習、10-01 以降を評価にする。
import implicit
import numpy as np
import pandas as pd
from scipy import sparse

from common import CODE, SEED, show

df = pd.read_parquet(CODE / "data" / "public" / "online_retail_ii.parquet")
df = df[~df.Invoice.str.startswith("C") & (df.Quantity > 0) & df["Customer ID"].notna()]
cut = pd.Timestamp("2011-10-01")
tr, ev = df[df.InvoiceDate < cut], df[df.InvoiceDate >= cut]
uid = {u: i for i, u in enumerate(sorted(tr["Customer ID"].unique()))}
iid = {s: i for i, s in enumerate(sorted(tr.StockCode.unique()))}
g = tr.groupby(["Customer ID", "StockCode"]).size().reset_index(name="n")
X = sparse.csr_matrix((np.log1p(g.n.to_numpy()).astype(np.float32), (g["Customer ID"].map(uid), g.StockCode.map(iid))),
                      shape=(len(uid), len(iid)))
truth = {uid[u]: set(iid[s] for s in grp.StockCode if s in iid) for u, grp in ev.groupby("Customer ID") if u in uid}
truth = {u: t for u, t in truth.items() if t}
users = np.array(sorted(truth))
print(f"学習 {len(tr):,} 行（会員 {len(uid):,}・商品 {len(iid):,}）、評価の会員 {len(users):,} 人")
K, rows = 100, []


def rec(c):
    return {f"Recall@{k}": float(np.mean([len(set(c[u][:k]) & truth[u]) / len(truth[u]) for u in users])) for k in (20, 50, 100)}


# %% Step 2 — 人気と共起｜直近 30 日の購入の多い順、同じ会員が買った商品どうしのコサイン類似度（会員の購入した商品から足す）。
last = tr[tr.InvoiceDate >= cut - pd.Timedelta(days=30)].StockCode.value_counts()
pop = [iid[s] for s in last.index if s in iid][:K]
rows.append({"方法": "人気（直近 30 日）", **rec({u: pop for u in users})})
B = (X > 0).astype(np.float32)
C = (B.T @ B).toarray(); np.fill_diagonal(C, 0)
deg = np.sqrt(np.asarray(B.sum(0)).ravel()) + 1e-9
sc = (B[users] @ (C / deg[:, None] / deg[None, :]))
rows.append({"方法": "共起（アイテム間の類似）", **rec({u: list(np.argsort(-sc[k])[:K]) for k, u in enumerate(users)})})

# %% Step 3 — 行列分解｜implicit の ALS と BPR（第9章の合成データと同じ設定）。
for name, M in (("ALS（implicit）", implicit.cpu.als.AlternatingLeastSquares(factors=64, regularization=0.05, iterations=15, random_state=SEED)),
                ("BPR（implicit）", implicit.cpu.bpr.BayesianPersonalizedRanking(factors=64, iterations=100, random_state=SEED))):
    M.fit(X, show_progress=False)
    ids, _ = M.recommend(users, X[users], N=K, filter_already_liked_items=False)
    rows.append({"方法": name, **rec({u: list(r) for u, r in zip(users, ids)})})

# %% Step 4 — 比較｜実データでの順位を、合成データ（01）の順位と見比べる。
show(pd.DataFrame(rows).set_index("方法"), "実データでの候補生成（UCI Online Retail II、評価 2011-10-01〜12-09）")
