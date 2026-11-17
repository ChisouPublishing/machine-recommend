"""第4章：オフライン方策評価。ログ（人気＋ε の方策）だけで、新しい方策のクリック率を見積もり、真の値と比べる。

対象：トップのおすすめの 1 番目の枠（位置の影響を分けるため）。検証期間 day 133〜146。
新しい方策 π_e を 2 つ用意する。A：会員のカテゴリの好み × 商品の人気、B：A にブランドの好みを足したもの。
どちらも 10% は一様に選ぶ（どの商品も選ばれうるようにする）。
"""
# %% Step 1 — ログを読む｜1 番目の枠の表示・クリック・傾向スコア（ログの方策 π_b がその商品を選んだ確率）。
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common import DATA, SEED, connect, show

sys.path.insert(0, str(DATA.parents[1]))
from truth import Truth  # noqa: E402  答え合わせ用（真の値の計算だけに使う）

con = connect()
D = "date_diff('day', DATE '2025-04-07', CAST({} AS DATE))"
log = con.sql(f"""
SELECT i.impression_id, i.session_id, i.user_id, u.gender, i.item_id, it.category_id - 1 AS cat,
       {D.format('i.shown_at')} AS day, i.propensity AS pb,
       EXISTS (SELECT 1 FROM event e WHERE e.impression_id = i.impression_id AND e.event_type = 'view')::INT AS r
FROM impression i JOIN "user" u USING (user_id) JOIN item it USING (item_id)
WHERE i.surface = 'top' AND i.position = 1 AND {D.format('i.shown_at')} BETWEEN 28 AND 146 ORDER BY i.impression_id
""").df()
cats = con.sql("SELECT COUNT(*) FROM category").fetchall()[0][0]
n_items = con.sql("SELECT COUNT(*) FROM item").fetchall()[0][0]
item_cat = con.sql("SELECT category_id - 1 FROM item ORDER BY item_id").df().iloc[:, 0].to_numpy()
item_gender = con.sql("SELECT gender FROM item ORDER BY item_id").df().iloc[:, 0].to_numpy(dtype=object)

# %% Step 2 — 方策の材料｜会員×日のカテゴリ別・ブランド別の閲覧数（前日までの 28 日）と、商品×日の人気（前日までの 7 日の購入数）。
cc = con.sql(f"""
WITH k AS (SELECT DISTINCT user_id, day FROM log)
SELECT k.user_id, k.day, it.category_id - 1 AS cat, COUNT(*) AS n
FROM k JOIN event e ON e.user_id = k.user_id AND e.event_type = 'view'
  AND {D.format('e.occurred_at')} BETWEEN k.day - 28 AND k.day - 1
JOIN item it ON it.item_id = e.item_id GROUP BY ALL""").df()
key = log[["user_id", "day"]].drop_duplicates().reset_index(drop=True)
key["k"] = np.arange(len(key))
catcnt = np.zeros((len(key), cats))
cc = cc.merge(key, on=["user_id", "day"])
catcnt[cc.k, cc.cat] = cc.n
log = log.merge(key, on=["user_id", "day"])
con.register("key", key)
bc = con.sql(f"""SELECT k.k, it.brand_id - 1 AS b, COUNT(*) AS n FROM key k
JOIN event e ON e.user_id = k.user_id AND e.event_type = 'view' AND {D.format('e.occurred_at')} BETWEEN k.day - 28 AND k.day - 1
JOIN item it ON it.item_id = e.item_id GROUP BY ALL""").df()
n_brands = con.sql("SELECT COUNT(*) FROM brand").fetchall()[0][0]
brcnt = np.zeros((len(key), n_brands))
brcnt[bc.k, bc.b] = bc.n
item_brand = con.sql("SELECT brand_id - 1 FROM item ORDER BY item_id").df().iloc[:, 0].to_numpy()
pp = con.sql(f"""SELECT item_id - 1 AS i, {D.format('occurred_at')} AS d, COUNT(*) AS n FROM event
                 WHERE event_type = 'purchase' GROUP BY ALL""").df()
daily = np.zeros((n_items, 184))   # 深夜をまたいだ記録が day 182 に入ることがある
daily[pp.i, pp.d] = pp.n
pop7 = np.zeros_like(daily)
for d in range(7, 184):
    pop7[:, d] = daily[:, d - 7:d].sum(1)
logpop = np.log1p(pop7)
avail = con.sql(f"SELECT item_id - 1 AS i, {D.format('date')} AS d FROM item_availability").df()
pools = {}
for (d), g in avail.groupby("d"):
    for gen in ["M", "F"]:
        ii = g.i.to_numpy()
        pools[(d, gen)] = ii[(item_gender[ii] == gen) | (item_gender[ii] == "U")]

# %% Step 3 — 新しい方策 π_e｜点数 s = β1·log(1+カテゴリの閲覧数) + βb·log(1+ブランドの閲覧数) + β2·log(1+人気)。確率は ε/候補数 ＋ (1−ε)·softmax(s)。
POLICIES = {"A：カテゴリ×人気": (1.0, 0.0, 0.5), "B：カテゴリ＋ブランド×人気": (1.0, 1.0, 0.5)}
EPS = 0.10


def pe_all(rows, beta):
    """rows（同じ日・同じ性別）について、候補の全商品の π_e を (行数, 候補数) で返す。"""
    b1, bb, b2 = beta
    d, g = int(rows.day.iloc[0]), rows.gender.iloc[0]
    pool = pools[(d, g)]
    k = rows.k.to_numpy()
    s = (b1 * np.log1p(catcnt[k][:, item_cat[pool]]) + bb * np.log1p(brcnt[k][:, item_brand[pool]])
         + b2 * logpop[pool, d][None, :])
    e = np.exp(s - s.max(1, keepdims=True))
    return pool, EPS / len(pool) + (1 - EPS) * e / e.sum(1, keepdims=True)


# %% Step 4 — 報酬のモデル q｜学習期間（day 28〜132）のログで、クリックを方策と同じ 3 つの材料から予測する（DR に使う）。
def feat(r):
    k, i = r.k.to_numpy(), r.item_id.to_numpy() - 1
    return np.column_stack([np.log1p(catcnt[k, item_cat[i]]), np.log1p(brcnt[k, item_brand[i]]), logpop[i, r.day.to_numpy()]])


tr = log[log.day <= 132]
qm = LogisticRegression().fit(feat(tr), tr.r)


def q_all(rows, pool):
    k, d = rows.k.to_numpy(), int(rows.day.iloc[0])
    z = (qm.intercept_[0] + qm.coef_[0, 0] * np.log1p(catcnt[k][:, item_cat[pool]])
         + qm.coef_[0, 1] * np.log1p(brcnt[k][:, item_brand[pool]]) + qm.coef_[0, 2] * logpop[pool, d][None, :])
    return 1 / (1 + np.exp(-z))


val = log[(log.day >= 133) & (log.day <= 146)].copy()
val["q"] = qm.predict_proba(feat(val))[:, 1]
groups = list(val.groupby(["day", "gender"]))


def weights(beta):
    """各ログの π_e(a)/π_b(a) と、DR の Σ_a π_e(a) q(x,a)。"""
    w, qpe = np.zeros(len(val)), np.zeros(len(val))
    loc = {ix: j for j, ix in enumerate(val.index)}
    for _, rows in groups:
        pool, P = pe_all(rows, beta)
        pos = {a: j for j, a in enumerate(pool)}
        idx = np.array([pos[a - 1] for a in rows.item_id])
        j = np.array([loc[ix] for ix in rows.index])
        w[j] = P[np.arange(len(rows)), idx] / rows.pb.to_numpy()
        qpe[j] = (P * q_all(rows, pool)).sum(1)
    return w, qpe


# %% Step 5 — 推定量｜IPS・SNIPS・DR を最小実装で計算し、ブートストラップで 95% 区間を付ける。
rng = np.random.default_rng(SEED)
B = [rng.integers(0, len(val), len(val)) for _ in range(200)]
r, q = val.r.to_numpy(), val.q.to_numpy()
EST = {"IPS": lambda w, qpe, i: (w[i] * r[i]).mean(),
       "SNIPS": lambda w, qpe, i: (w[i] * r[i]).sum() / w[i].sum(),
       "DR": lambda w, qpe, i: (qpe[i] + w[i] * (r[i] - q[i])).mean()}
rows_out, cache = [], {}
allidx = np.arange(len(val))
for name, beta in POLICIES.items():
    w, qpe = cache[name] = weights(beta)
    for en, f in EST.items():
        lo, hi = np.quantile([f(w, qpe, b) for b in B], [0.025, 0.975])
        rows_out.append((name, en, f(w, qpe, allidx), lo, hi))
    print(f"{name}: 重み w = π_e/π_b の平均 {w.mean():.3f}・最大 {w.max():.1f}")

# %% Step 6 — 真の値｜状態を記録したセッション（約 5%）で、候補の全商品の真のクリック確率から各方策の価値を計算する。
T = Truth(DATA)
sub = val[val.session_id.isin(T.state.index)]
truth = {}
for name, beta in POLICIES.items():
    tv = []
    for _, rows in sub.groupby(["day", "gender"]):
        pool, P = pe_all(rows, beta)
        pc = T.p_click(rows.session_id.to_numpy(), np.broadcast_to(pool + 1, P.shape), np.ones(P.shape))
        tv.append((P * pc).sum(1))
    tv = np.concatenate(tv)
    truth[name] = (tv.mean(), tv.std(ddof=1) / np.sqrt(len(tv)))
tb = connect(truth=True).sql(f"""SELECT AVG(t.p_click) FROM impression i JOIN truth_impression t USING (impression_id)
       WHERE i.surface = 'top' AND i.position = 1 AND {D.format('i.shown_at')} BETWEEN 133 AND 146""").fetchall()[0][0]
print(f"検証の 1 番目の枠: {len(val):,} 件。ログの方策 π_b の真の価値 {tb:.4f}、ログのクリック率 {r.mean():.4f}")
for name, (m, se) in truth.items():
    print(f"{name} の真の価値: {m:.4f}（{len(sub):,} セッション、標準誤差 {se:.4f}）")
show(pd.DataFrame(rows_out, columns=["方策", "推定量", "推定値", "95%下限", "95%上限"]),
     "π_e のクリック率の推定（1 番目の枠、検証 day 133〜146、合成データ）", index=False)

# %% Step 7 — 発展版：Open Bandit Pipeline｜同じログの一部（3,000 件）で、obp の推定量が最小実装と一致するかを確かめる。
from obp.ope import DoublyRobust, InverseProbabilityWeighting, OffPolicyEvaluation, SelfNormalizedInverseProbabilityWeighting

idx = np.sort(rng.choice(len(val), 3000, replace=False))
part = val.iloc[idx]
name, beta = "B：カテゴリ＋ブランド×人気", POLICIES["B：カテゴリ＋ブランド×人気"]
dist = np.zeros((len(part), n_items, 1))
qall = np.zeros((len(part), n_items, 1))
for _, rows in part.groupby(["day", "gender"]):
    pool, P = pe_all(rows, beta)
    j = np.array([part.index.get_loc(ix) for ix in rows.index])
    dist[j[:, None], pool[None, :], 0] = P
    qall[j[:, None], pool[None, :], 0] = q_all(rows, pool)
fb = {"n_rounds": len(part), "n_actions": n_items, "action": part.item_id.to_numpy() - 1, "reward": part.r.to_numpy(),
      "pscore": part.pb.to_numpy(), "position": np.zeros(len(part), dtype=int)}
ope = OffPolicyEvaluation(bandit_feedback=fb, ope_estimators=[InverseProbabilityWeighting(),
                          SelfNormalizedInverseProbabilityWeighting(), DoublyRobust()])
obp_v = ope.estimate_policy_values(action_dist=dist, estimated_rewards_by_reg_model=qall)
w, qpe = cache[name]
mine = {"ipw": (w[idx] * r[idx]).mean(), "snipw": (w[idx] * r[idx]).sum() / w[idx].sum(),
        "dr": (qpe[idx] + w[idx] * (r[idx] - q[idx])).mean()}
show(pd.DataFrame({"obp": obp_v, "最小実装": mine}).rename(index={"ipw": "IPS", "snipw": "SNIPS", "dr": "DR"}),
     f"obp と最小実装（{name}、3,000 件）", digits=6)
