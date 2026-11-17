"""第11章：統一ベンチマーク。第5〜8章の候補 10 個を、同じ期間・間引き・校正・指標で、乱数の種 3 つずつ評価する。"""
# %% Step 1 — 準備｜候補ごと・種ごとの予測を保存し、既定では保存したものを読む（作り直しは run_all.py --search）。
import os

import numpy as np
import pandas as pd

from common import OUT, auc, ece, logloss, save_table, show
from common.bench import CANDS, PERIOD, calibrate, rest_of, run
from common.features import load_top

SEEDS = (0, 1, 2)
df = load_top()
df = df[df.day <= 146]                                       # 開発期間だけ（テストは読まない）
rest = rest_of(df)
ev = rest[rest.day >= PERIOD["eval"][0]]
y = ev.purchased.to_numpy()
store = OUT / "11_selection" / "bench"
store.mkdir(parents=True, exist_ok=True)
redo = os.environ.get("SUISEN_SEARCH") == "1"
preds, p99 = {}, {}
for name in CANDS:
    ps, ts = [], []
    for s in SEEDS:
        f = store / f"{name}_{s}.npz"
        if redo or not f.exists():
            p_all, t = run(name, df, s)
            np.savez(f, p=calibrate(p_all, rest), p99=t)
        z = np.load(f)
        ps.append(z["p"]); ts.append(float(z["p99"]))
    preds[name], p99[name] = ps, ts

# %% Step 2 — 業務の指標｜期待粗利（校正後の確率 × 1 件の返品控除後の粗利）の上位 10% の表示で、実際に得た返品控除後の粗利を数える。
from common import connect

con = connect()
it = con.sql("SELECT item_id, list_price - unit_cost AS margin FROM item").df()
rr = con.sql("""SELECT it.category_id, AVG((r.return_id IS NOT NULL)::INT) AS ret FROM order_line l JOIN item it USING (item_id)
 JOIN "order" o USING (order_id) LEFT JOIN "return" r USING (order_line_id)
 WHERE date_diff('day', DATE '2025-04-07', CAST(o.ordered_at AS DATE)) <= 118 GROUP BY 1""").df()
real = con.sql("""SELECT l.impression_id, CASE WHEN r.return_id IS NULL THEN l.unit_price - l.unit_cost ELSE -1000 END AS gp
 FROM order_line l LEFT JOIN "return" r USING (order_line_id) WHERE l.source_surface = 'top'""").df()
e = ev[["impression_id", "item_id", "category_id"]].merge(it, on="item_id", how="left").merge(rr, on="category_id", how="left")
value = (e.margin * (1 - e.ret) - e.ret * 1000).to_numpy()
gp = ev[["impression_id"]].merge(real, on="impression_id", how="left").gp.fillna(0).to_numpy()


def profit(p):
    o = np.argsort(-(p * value))[: int(len(p) * 0.1)]
    return gp[o].sum()


# %% Step 3 — 比較の表｜種ごとの指標の平均（と LogLoss の標準偏差）、P99（種の中央値）、解釈のしやすさ。
rows = []
for name, ps in preds.items():
    m = [(logloss(y, p), auc(y, p), ece(y, p), profit(p)) for p in ps]
    ll, au, ec, pr = map(np.array, zip(*m))
    fam, impl, interp = CANDS[name]
    rows.append({"候補": name, "実装": impl, "LogLoss": ll.mean(), "LogLoss 標準偏差": ll.std(ddof=1), "AUC": au.mean(),
                 "ECE": ec.mean(), "期待粗利（千円）": pr.mean() / 1000, "P99（ms）": float(np.median(p99[name])), "解釈": interp})
tab = pd.DataFrame(rows).sort_values("LogLoss").reset_index(drop=True)
show(tab, "統一ベンチマーク（評価 day 133〜146、校正後、種 3 つの平均、合成データ）", index=False, digits=5)
save_table(tab, "11_selection", "benchmark", digits=5)

# %% Step 4 — 同じ条件の確かめ｜全候補が同じ表（行数・ハッシュ）・同じ期間で評価されたことと、各章の探索の試行数・時間を残す。
from common.tuning import counts, get_study

key = df[["impression_id", "day", "purchased"]].sort_values("impression_id")
h = format(int(pd.util.hash_pandas_object(key, index=False).sum()) & 0xFFFFFFFFFFFF, "012x")
print(f"全候補で共通：開発期間の表 {len(df):,} 行（ハッシュ {h}）、期間 {PERIOD}、校正 Platt、種 {SEEDS}")
STUDY = {"LR": "05_lr", "LightGBM": "06_gbdt_lgbm", "XGBoost": "06_gbdt_xgb", "CatBoost": "06_gbdt_cat", "FM（myFM）": "07_fm_myfm",
         "DeepFM（最小実装）": "08_dnn_mine_DeepFM", "DeepFM（FuxiCTR）": "08_dnn_fuxi_DeepFM", "DCN-V2（FuxiCTR）": "08_dnn_fuxi_DCNv2",
         "DIN（最小実装）": "08_dnn_mine_DIN", "DIN（FuxiCTR）": "08_dnn_fuxi_DIN"}
sr = []
for name in tab["候補"]:
    c = counts(get_study(STUDY[name]))
    sr.append({"候補": name, "study": STUDY[name], "完了": c["完了"], "枝刈り": c["枝刈り"], "探索（分、保存時の記録）": c["探索秒"] / 60})
show(pd.DataFrame(sr), "各候補の構成を決めた探索（Optuna の履歴から）", index=False, digits=1)

# %% Step 5 — 不確かさ｜種の平均の予測で、評価の行をブートストラップ（200 回）し、最良の候補との LogLoss の差の 95% 区間と、
#               差が 0 以下の割合から求めた p 値を Holm 法で補正する（9 つの比較）。
rng = np.random.default_rng(0)
B = [rng.integers(0, len(y), len(y)) for _ in range(200)]
mean_p = {k: np.mean(v, axis=0) for k, v in preds.items()}
top = tab.iloc[0]["候補"]
ll_b = lambda p, i: logloss(y[i], p[i])
base = np.array([ll_b(mean_p[top], i) for i in B])
cmp = []
for name in tab["候補"][1:]:
    d = np.array([ll_b(mean_p[name], i) for i in B]) - base
    lo, hi = np.quantile(d, [0.025, 0.975])
    cmp.append((name, d.mean(), lo, hi, min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))))
c = pd.DataFrame(cmp, columns=["候補", "LogLoss の差（対 最良）", "95%下限", "95%上限", "p 値"]).sort_values("p 値").reset_index(drop=True)
c["Holm 補正後"] = np.minimum(1, np.maximum.accumulate(c["p 値"] * (len(c) - np.arange(len(c)))))
show(c.sort_values("LogLoss の差（対 最良）"), f"最良（{top}）との差（評価の行のブートストラップ 200 回）", index=False, digits=5)
save_table(c, "11_selection", "differences", digits=5)
