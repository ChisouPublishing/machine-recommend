"""第11章：実データ（UCI Online Retail II、CC BY 4.0）で、選定の順位（LR より GBDT が良いか）が再現するかを確かめる。

課題：2011-09-01 の時点で、会員×商品の候補（共起の上位 50 件）について、その後 60 日に買うかを予測する。
出典：Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D
"""
# %% Step 1 — データ｜取消・数量 0 以下・会員番号なしを除く。2011-06-01 と 09-01 の 2 つの時点で、学習と評価の表を作る。
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import CODE, SEED, auc, logloss, show

raw = pd.read_parquet(CODE / "data" / "public" / "online_retail_ii.parquet")
raw = raw[~raw.Invoice.str.startswith("C") & (raw.Quantity > 0) & raw["Customer ID"].notna()]


def table(cut: pd.Timestamp, k: int = 50) -> pd.DataFrame:
    past, fut = raw[raw.InvoiceDate < cut], raw[(raw.InvoiceDate >= cut) & (raw.InvoiceDate < cut + pd.Timedelta(days=60))]
    users, items = sorted(past["Customer ID"].unique()), sorted(past.StockCode.unique())
    ui, ii = {u: i for i, u in enumerate(users)}, {s: i for i, s in enumerate(items)}
    B = sparse.csr_matrix((np.ones(len(past)), (past["Customer ID"].map(ui), past.StockCode.map(ii))), shape=(len(users), len(items)))
    B.data[:] = 1
    C = (B.T @ B).toarray(); np.fill_diagonal(C, 0)
    deg = np.sqrt(np.asarray(B.sum(0)).ravel()) + 1e-9
    S = B @ (C / deg[:, None] / deg[None, :])
    top = np.argsort(-S, axis=1)[:, :k]
    rows = pd.DataFrame({"u": np.repeat(np.arange(len(users)), k), "i": top.ravel()})
    rows["co"] = S[rows.u, rows.i]
    last = past.groupby(["Customer ID", "StockCode"]).InvoiceDate.max()
    n_ui = past.groupby(["Customer ID", "StockCode"]).size()
    key = list(zip(np.array(users)[rows.u], np.array(items)[rows.i]))
    rows["bought_before"] = [n_ui.get(t, 0) for t in key]
    rows["days_since"] = [(cut - last[t]).days if t in last.index else 999 for t in key]
    pop = past[past.InvoiceDate >= cut - pd.Timedelta(days=30)].StockCode.value_counts()
    rows["item_pop30"] = [pop.get(items[i], 0) for i in rows.i]
    freq = past.groupby("Customer ID").Invoice.nunique()
    rows["user_orders"] = [freq[users[u]] for u in rows.u]
    price = past.groupby("StockCode").Price.median()
    rows["price"] = [price[items[i]] for i in rows.i]
    got = set(zip(fut["Customer ID"], fut.StockCode))
    rows["y"] = [int(t in got) for t in key]
    return rows


F = ["co", "bought_before", "days_since", "item_pop30", "user_orders", "price"]
tr, te = table(pd.Timestamp("2011-06-01")), table(pd.Timestamp("2011-09-01"))
print(f"学習 {len(tr):,} 組（購入率 {tr.y.mean():.4f}）・評価 {len(te):,} 組（購入率 {te.y.mean():.4f}）")

# %% Step 2 — 2 つの候補｜LR（log(1+x) と標準化）と LightGBM。合成データの選定と同じく、GBDT が LR より良いかを見る。
X = lambda d: np.log1p(d[F].clip(lower=0))
sc = StandardScaler().fit(X(tr))
p_lr = LogisticRegression(max_iter=1000).fit(sc.transform(X(tr)), tr.y).predict_proba(sc.transform(X(te)))[:, 1]
p_gb = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=SEED, verbose=-1,
                          deterministic=True, n_jobs=1).fit(tr[F], tr.y).predict_proba(te[F])[:, 1]
show(pd.DataFrame({"AUC": {"LR": auc(te.y, p_lr), "LightGBM": auc(te.y, p_gb)},
                   "LogLoss": {"LR": logloss(te.y, p_lr), "LightGBM": logloss(te.y, p_gb)}}),
     "実データでの再現（UCI Online Retail II、2011-09-01 から 60 日）", digits=4)
