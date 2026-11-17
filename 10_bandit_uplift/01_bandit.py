"""第10章：新着の枠のバンディット。27 のカテゴリから 1 つを選び、そのカテゴリの新着で最も人気の商品を出す。真のクリック確率で報酬を引く。"""
# %% Step 1 — 環境｜状態を記録したセッション（day 28〜181、約 2.6 万）を順に到着させる。文脈は会員のカテゴリ別の閲覧数（前日までの 28 日）。
import sys

import numpy as np
import pandas as pd

from common import DATA, SEED, connect, savefig, show
from common.models.bandit import EpsGreedy, LinTS, LinUCB

sys.path.insert(0, str(DATA.parents[1]))
from truth import Truth  # noqa: E402  報酬を引くための真の確率

T = Truth(DATA)
con = connect()
D = "date_diff('day', DATE '2025-04-07', CAST({} AS DATE))"
st = T.state.reset_index()
st = st[st.day >= 28].sort_values(["day", "session_id"]).reset_index(drop=True)
con.register("st", st[["session_id", "user_id", "day"]])
cc = con.sql(f"""SELECT st.session_id, it.category_id - 1 AS c, COUNT(*) AS n FROM st JOIN event e ON e.user_id = st.user_id
 AND e.event_type = 'view' AND {D.format('e.occurred_at')} BETWEEN st.day - 28 AND st.day - 1 JOIN item it USING (item_id) GROUP BY ALL""").df()
pos = {s: i for i, s in enumerate(st.session_id)}
X = np.zeros((len(st), 27))
X[cc.session_id.map(pos), cc.c] = cc.n
X = np.log1p(X)
X = np.column_stack([X / (X.sum(1, keepdims=True) + 1e-9), np.ones(len(st))])   # カテゴリの好みの割合 ＋ 定数
gender = con.sql('SELECT user_id, gender FROM "user"').df().set_index("user_id").gender

# %% Step 2 — 腕と商品｜日・性別・カテゴリごとに、出せる新着（発売 30 日以内）で前日までの 7 日の購入が最も多い商品。新着が無ければ既存の最も人気の商品。
it = con.sql("SELECT item_id, category_id - 1 AS c, gender, date_diff('day', DATE '2025-04-07', release_date) AS rd FROM item").df()
av = con.sql(f"SELECT item_id, {D.format('date')} AS d FROM item_availability WHERE {D.format('date')} >= 28").df()
pp = con.sql(f"""SELECT item_id, {D.format('occurred_at')} AS d, COUNT(*) AS n FROM event WHERE event_type = 'purchase' GROUP BY ALL""").df()
daily = np.zeros((8001, 190)); daily[pp.item_id, pp.d] = pp.n
arm_item = {}
for d, g in av.merge(it, on="item_id").groupby("d"):
    pop = daily[g.item_id.to_numpy(), max(d - 7, 0):d].sum(1)
    g = g.assign(pop=pop, new=(d - g.rd) <= 30)
    for gen in ("M", "F"):
        gg = g[g.gender.isin([gen, "U"])].sort_values(["new", "pop", "item_id"], ascending=[False, False, True])
        first = gg.drop_duplicates("c")
        arm_item[(d, gen)] = np.array([first.set_index("c").item_id.get(c, first.item_id.iloc[0]) for c in range(27)])
items = np.stack([arm_item[(d, gender[u])] for d, u in zip(st.day, st.user_id)])
P = T.p_click(st.session_id.to_numpy(), items, np.ones(items.shape))            # 各回の 27 腕の真のクリック確率
print(f"{len(st):,} 回、腕 27、真のクリック確率：最良の腕の平均 {P.max(1).mean():.4f}・全腕の平均 {P.mean():.4f}")

# %% Step 3 — 方策を走らせる｜報酬は真の確率で引く（同じ乱数で全方策を比べる）。累積の後悔 = Σ（最良の腕の確率 − 選んだ腕の確率）。
rng = np.random.default_rng(SEED)
U = rng.random(len(st))
POL = {"一様ランダム": None, "ε-greedy（ε=0.1）": EpsGreedy(27, 0.1, SEED), "LinUCB（α=0.5、最小実装）": LinUCB(27, 28, 0.5),
       "線形 TS（最小実装）": LinTS(27, 28, 0.25, SEED)}
regret, clicks = {}, {}
for name, pol in POL.items():
    reg, cl = np.zeros(len(st)), 0
    for t in range(len(st)):
        a = int(rng.integers(27)) if pol is None else pol.select(X[t])
        r = float(U[t] < P[t, a])
        if pol is not None:
            pol.update(a, X[t], r)
        reg[t], cl = P[t].max() - P[t, a], cl + r
    regret[name], clicks[name] = np.cumsum(reg), cl / len(st)

# %% Step 4 — 発展版：Open Bandit Pipeline｜obp の LinUCB・LinTS を同じ環境で走らせる（腕ごとの線形モデル、更新は 1 回ずつ）。
from obp.policy import LinTS as ObpLinTS, LinUCB as ObpLinUCB

for name, pol in (("LinUCB（obp）", ObpLinUCB(dim=28, n_actions=27, epsilon=0.5, random_state=SEED)),
                  ("線形 TS（obp）", ObpLinTS(dim=28, n_actions=27, random_state=SEED))):
    reg, cl = np.zeros(len(st)), 0
    for t in range(len(st)):
        a = int(pol.select_action(X[t][None, :])[0])
        r = float(U[t] < P[t, a])
        pol.update_params(action=a, reward=r, context=X[t][None, :])
        reg[t], cl = P[t].max() - P[t, a], cl + r
    regret[name], clicks[name] = np.cumsum(reg), cl / len(st)

# %% Step 5 — 比較｜累積の後悔（最後の値）とクリック率。図は累積の後悔の推移。
show(pd.DataFrame({"累積の後悔": {k: v[-1] for k, v in regret.items()}, "クリック率": clicks}),
     f"新着の枠のバンディット（{len(st):,} 回、合成データ）", digits=4)
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4))
for k, v in regret.items():
    ax.plot(v, label=k, lw=1.2)
ax.set(xlabel="回（到着したセッション）", ylabel="累積の後悔")
ax.legend(fontsize=7)
savefig(fig, "10_bandit_uplift", "regret")
