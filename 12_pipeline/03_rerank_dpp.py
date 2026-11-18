"""第12章 12.4：再ランキングの多様性。MMR と DPP（高速な貪欲法）を、同じ候補・同じ点数・同じブランドの上限で比べる。"""
# %% Step 1 — 準備｜01 と同じ ALS の候補 100 件を、同じ軽いランキングで採点する（在庫切れは各方法の中で除く）。
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import savefig, show
from common.pipeline import Pipeline

P = Pipeline()
pairs = P.score(P.cands["ALS（implicit）"])
groups = [(s, g) for s, g in pairs.groupby("session_id", sort=False)]
IC, IB = P.IC, P.IB


# %% Step 2 — MMR｜01 と同じ。点数（最大を 1 に）と、選んだ商品との似ている度合い（カテゴリが同じ 0.5 ＋ブランドが同じ 0.5）の最大を、λ で混ぜて 1 件ずつ選ぶ。
def mmr(g, lam=0.7, brand_cap=2):
    g = g[g.ok]
    ids = g.item_id.to_numpy()
    sc = g.score.to_numpy() / g.score.max()
    c, b = IC[ids - 1], IB[ids - 1]
    sim = np.zeros(len(ids))
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


# %% Step 3 — DPP｜核 L_ij = q_i S_ij q_j。q_i = exp(α·log 点数) が良さ、S_ij（同じ似ている度合い。対角は 1）が似ていなさを表す。
#               選んだ集合の行列式が最も増える商品を 1 件ずつ足す（Chen ほか, 2018 の高速な貪欲法。コレスキー分解を 1 行ずつ更新）。
#               α = θ / (2(1 − θ))。θ を 1 に近づけるほど点数を、0 に近づけるほど多様性を重視する。
def dpp(g, theta=0.7, brand_cap=2, eps=1e-10):
    g = g[g.ok]
    ids = g.item_id.to_numpy()
    n = len(ids)
    c, b = IC[ids - 1], IB[ids - 1]
    q = np.exp(theta / (2 * (1 - theta)) * np.log(g.score.to_numpy()))
    S = 0.5 * (c[:, None] == c[None, :]) + 0.5 * (b[:, None] == b[None, :])   # 0.5·(カテゴリの one-hot) と 0.5·(ブランドの one-hot) の内積なので半正定値
    L = q[:, None] * S * q[None, :]
    d2 = np.diag(L).copy()                                     # 選んだ集合に足したときの行列式の増え方（の残り）
    C = np.zeros((6, n))
    chosen, cnt = [], {}
    for t in range(6):
        cand = d2.copy()
        cand[chosen] = -np.inf
        cand[[i for i in range(n) if cnt.get(b[i], 0) >= brand_cap]] = -np.inf
        j = int(np.argmax(cand))
        if cand[j] < eps:                                      # もう行列式を増やせない（似た商品しか残っていない）→ 点数の順で埋める
            rest = [i for i in np.argsort(-q, kind="stable") if i not in chosen and cnt.get(b[i], 0) < brand_cap]
            chosen += rest[:6 - len(chosen)]
            break
        chosen.append(j); cnt[b[j]] = cnt.get(b[j], 0) + 1
        e = (L[j] - C[:t].T @ C[:t, j]) / np.sqrt(d2[j])
        C[t] = e
        d2 = d2 - e ** 2
    return list(ids[chosen])


# %% Step 4 — 比べる｜MMR の λ と DPP の θ を 4 段ずつ変え、期待クリック（真の確率）とブランド・カテゴリの数、1 セッションあたりの時間を測る。
rows, curve = {}, {"MMR": [], "DPP": []}
base = {}
for s, g in groups:
    gs = g.sort_values("score", ascending=False)                         # 01 の (c) と同じ並べ方
    base[s] = list(gs[gs.ok].item_id[:6])
rows["点数の順（在庫切れを除く）"] = {**P.evaluate(base), "ms／セッション": np.nan}
for name, fn, vals in (("MMR", mmr, (0.9, 0.7, 0.5, 0.3)), ("DPP", dpp, (0.9, 0.7, 0.5, 0.3))):
    for v in vals:
        t = time.perf_counter()
        sel = {s: fn(g, v) for s, g in groups}
        ms = (time.perf_counter() - t) / len(groups) * 1000
        r = P.evaluate(sel)
        rows[f"{name}（{'λ' if name == 'MMR' else 'θ'}={v}）"] = {**r, "ms／セッション": ms}
        curve[name].append((r["カテゴリの数"], r["期待クリック数（6 枠）"], v))
tab = pd.DataFrame(rows).T.drop(columns=["在庫切れ（件）"])
for k, v in tab["ms／セッション"].dropna().items():
    print(f"[time] {k} 1 セッションあたり {v:.2f} ms")
show(tab.drop(columns=["ms／セッション"]), f"MMR と DPP（{len(groups):,} セッション、同じブランドは 2 件まで、合成データ）")

# %% Step 5 — 図｜横軸にカテゴリの数、縦軸に期待クリックを取り、2 つの方法の取引の曲線を重ねる（右上ほど良い）。
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for name, mk in (("MMR", "o"), ("DPP", "s")):
    x, y, v = zip(*curve[name])
    ax.plot(x, y, marker=mk, label=name)
    for xi, yi, vi in zip(x, y, v):
        ax.annotate(f"{'λ' if name == 'MMR' else 'θ'}={vi}", (xi, yi), textcoords="offset points", fontsize=8,
                    xytext=(6, 6) if name == "MMR" else (-8, -14), ha="left" if name == "MMR" else "right")   # 2 つのラベルが重ならないよう上下に分ける
r0 = rows["点数の順（在庫切れを除く）"]
ax.plot([r0["カテゴリの数"]], [r0["期待クリック数（6 枠）"]], "k*", ms=10, label="点数の順")
ax.set_xlabel("6 枠のカテゴリの数（平均）"); ax.set_ylabel("期待クリック数（6 枠、真の確率）")
ax.legend(); ax.grid(alpha=0.3)
savefig(fig, "12_pipeline", "mmr_dpp")
