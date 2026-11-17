"""第5章：Optuna で、ロジスティック回帰（LBFGS）の正則化の強さ C とハッシュの列数 2^bits を、時系列 CV で調整する。"""
# %% Step 1 — 準備｜3 折の時系列 CV（検証 day 105〜118・119〜132・133〜146）。ハッシュは一度だけ計算し、bits は剰余で変える。
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common import OUT, logloss, show
from common.data import cv_folds
from common.features import load_top, split
from common.models.linear import NumScaler, design, hash_codes
from common.tuning import counts, cv_objective, run_or_load, suggest, unprefix

df = load_top()
df = df[df.day <= 146]                                 # 開発期間だけ（テストは読まない）
codes = hash_codes(df)
folds = cv_folds()


def fit_score(params, fold):
    a, b = fold["train"]
    va_a, va_b = fold["valid"]
    tr_m, va_m = ((df.day >= a) & (df.day <= b)).to_numpy(), ((df.day >= va_a) & (df.day <= va_b)).to_numpy()
    tr, va = df[tr_m], df[va_m]
    sc = NumScaler().fit(tr)
    m = LogisticRegression(C=params["C"], max_iter=300)
    m.fit(design(tr, sc, codes[tr_m], params["bits"]), tr.purchased)
    return logloss(va.purchased, m.predict_proba(design(va, sc, codes[va_m], params["bits"]))[:, 1])


# %% Step 2 — 目的関数｜パラメーター名に「lr__」を付けて提案し、折ごとに LogLoss を報告する（悪い試行は途中で打ち切る）。
def objective(trial):
    p = {"C": suggest(trial, "lr", "C", "float", 1e-3, 10.0, log=True),
         "bits": suggest(trial, "lr", "bits", "int", 16, 22)}
    return cv_objective(trial, folds, lambda f, k: fit_score(p, f))


# %% Step 3 — 探索｜保存した履歴があれば読み、無ければ 25 試行を探索して保存する（run_all.py --search でやり直す）。
st = run_or_load("05_lr", objective, n_trials=25)
c = counts(st)
print("試行:", {k: v for k, v in c.items() if k != "探索秒"})
print(f"[time] 探索 {c['探索秒']} 秒")
best = unprefix(st.best_params, "lr")
print("最良の構成:", {k: round(v, 4) if isinstance(v, float) else v for k, v in best.items()}, f"CV の LogLoss {st.best_value:.5f}")

# %% Step 4 — 上位の試行｜LogLoss の小さい順に 5 つ並べる。差が小さければ、単純な（列の少ない）構成を選んでよい。
t = st.trials_dataframe(attrs=("number", "value", "params", "state"))
t = t[t.state == "COMPLETE"].sort_values("value").head(5)
t.columns = [c.replace("params_lr__", "") for c in t.columns]
show(t[["number", "C", "bits", "value"]].rename(columns={"number": "試行", "value": "CV の LogLoss"}), "上位 5 試行", index=False, digits=5)

# %% Step 5 — 測り直し｜最良の構成を折ごとに評価し直し、ばらつきを見る。比較の表（第11章）には、この平均と標準偏差を使う。
per = [fit_score(best, f) for f in folds]
se = np.std(per, ddof=1) / np.sqrt(len(per))
print("折ごとの LogLoss:", [round(v, 5) for v in per], f"平均 {np.mean(per):.5f}・標準誤差 {se:.5f}")

# %% Step 6 — 採用｜最良から標準誤差 1 つ以内の構成のうち、列の最も少ない（bits の小さい）ものを採る（単純さを優先する規則）。
ok = st.trials_dataframe(attrs=("value", "params", "state"))
ok = ok[(ok.state == "COMPLETE") & (ok.value <= st.best_value + se)].sort_values(["params_lr__bits", "value"])
adopt = {"C": float(ok.iloc[0]["params_lr__C"]), "bits": int(ok.iloc[0]["params_lr__bits"])}
print("採用する構成:", {"C": round(adopt["C"], 4), "bits": adopt["bits"]}, f"（標準誤差 1 つ以内の {len(ok)} 試行のうち最も単純）")
(OUT / "05_lr").mkdir(parents=True, exist_ok=True)
(OUT / "05_lr" / "best_lr.json").write_text(json.dumps(adopt))
