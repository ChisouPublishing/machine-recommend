"""第12章 12.3：複数の候補源を合わせる。ALS だけ・交互に取る・順位で合わせる（RRF）・新規会員の切り替えを、候補の再現率と多段の成果で比べる。"""
# %% Step 1 — 準備｜第9章の候補（人気・共起・ALS、two-tower は保存したベクトルから作る）と、01 と同じ評価のセッション・軽いランキング。
import time

import numpy as np
import pandas as pd

from common import show
from common.pipeline import D, Pipeline
from common.retrieval import TRAIN, eval_users, interactions, recall, true_recall, true_top

P = Pipeline()
src = P.cands
need = np.array(sorted(src["ALS（implicit）"]))
UV, IV = np.load("outputs/09_retrieval/user_vec.npy"), np.load("outputs/09_retrieval/item_vec.npy")
assert len(UV) == len(need)                                            # 第9章の 02 と同じ会員の並び
src["two-tower"] = {u: list(r) for u, r in zip(need, np.argsort(-(UV @ IV.T), 1)[:, :100] + 1)}
n_hist = interactions(*TRAIN).groupby("user_id").size()
new_user = {u: n_hist.get(u, 0) < 5 for u in need}                     # 学習期間の行動が 5 件未満を「新規会員」とする
print(f"会員 {len(need):,} 人（新規会員 {sum(new_user.values()):,} 人）")

# %% Step 2 — 新着の候補源｜day 119〜132 に発売された、会員の性別に合う商品を、直近 7 日の購入の多い順に。行動の少ない会員にも出せる。
rel = P.con.sql(f"SELECT item_id, gender, {D.format('release_date')} AS r FROM item WHERE {D.format('release_date')} BETWEEN 119 AND 132").df()
rel["pop"] = P.pop7[rel.item_id - 1, 132]
rel = rel.sort_values(["pop", "r", "item_id"], ascending=[False, False, True])
new_items = {g: [i for i, ig in zip(rel.item_id, rel.gender) if ig in (g, "U")][:100] for g in ("M", "F")}
src["新着"] = {u: new_items[P.gender[u]] for u in need}


# %% Step 3 — 合わせ方｜(1) 交互に取る：候補源を順に 1 件ずつ取り、重複を除いて 100 件。(2) RRF：候補源ごとの順位 r から 1/(60 + r) を足した点数の順。
#               (3) (2) に加え、新規会員は人気と新着を交互に取る候補に切り替える。
def interleave(lists, k=100):
    out, seen = [], set()
    for row in zip(*lists):
        for i in row:
            if i not in seen:
                seen.add(i); out.append(i)
    return out[:k]


def rrf(lists, k=100, c=60):
    sc = {}
    for l in lists:
        for r, i in enumerate(l):
            sc[i] = sc.get(i, 0.0) + 1.0 / (c + r + 1)
    return [i for i, _ in sorted(sc.items(), key=lambda t: (-t[1], t[0]))][:k]


MIX = ["ALS（implicit）", "共起（アイテム間の類似）", "two-tower", "人気（直近 7 日）"]
t = time.perf_counter()
sets = {"ALS だけ（01 と同じ）": src["ALS（implicit）"],
        "4 つを交互に取る": {u: interleave([src[s][u] for s in MIX]) for u in need},
        "4 つを RRF で合わせる": {u: rrf([src[s][u] for s in MIX]) for u in need}}
merge_ms = (time.perf_counter() - t) / len(need) / 2 * 1000
sets["RRF＋新規会員は人気・新着"] = {u: interleave([src["人気（直近 7 日）"][u], src["新着"][u]]) if new_user[u] else sets["4 つを RRF で合わせる"][u]
                                    for u in need}

# %% Step 4 — 候補の評価｜Recall@100（評価の期間の実際の行動、5,000 人）と、真の上位 20 件の再現率（1,500 セッション）を、新規会員とそれ以外に分けて。
users, truth = eval_users()
tt = true_top()
tt_new = tt[tt.user_id.map(new_user)]
tt_old = tt[~tt.user_id.map(new_user)]
rows = {}
for name, c in sets.items():
    rows[name] = {"Recall@100": recall({u: c[u] for u in users}, truth, ks=(100,))["Recall@100"],
                  "真の上位20@100": true_recall(c, tt)["真の上位20の再現率@100"],
                  "同（新規会員）": true_recall(c, tt_new)["真の上位20の再現率@100"],
                  "同（それ以外）": true_recall(c, tt_old)["真の上位20の再現率@100"]}
show(pd.DataFrame(rows).T, f"候補の比較（候補 100 件、新規会員のセッション {len(tt_new):,}、合成データ）")

# %% Step 5 — 多段の成果｜どの候補でも、01 の (c)（ランキング＋在庫切れを除く）で 6 枠を選び、真の期待クリックで比べる。
rows = {}
for name, c in sets.items():
    pairs = P.score(c)
    sel = {}
    for s, g in pairs.groupby("session_id", sort=False):
        gs = g.sort_values("score", ascending=False)                     # 01 と同じ並べ方（同点の扱いもそろえる）
        sel[s] = list(gs[gs.ok].item_id[:6])
    rows[name] = {**P.evaluate(sel), "期待クリック（新規会員）": P.evaluate(sel, tt_new.session_id)["期待クリック数（6 枠）"],
                  "期待クリック（それ以外）": P.evaluate(sel, tt_old.session_id)["期待クリック数（6 枠）"]}
show(pd.DataFrame(rows).T.drop(columns=["在庫切れ（件）"]), f"多段の成果（{len(P.st):,} セッション、ランキング＋在庫切れを除く、合成データ）")
print(f"[time] 候補を合わせる時間 1 人あたり {merge_ms:.3f} ms")
