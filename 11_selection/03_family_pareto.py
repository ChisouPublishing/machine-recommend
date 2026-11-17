"""第11章：Optuna で (1) モデル族そのものを探索対象にした study と、(2) 予測の良さと速さの 2 つの目的（NSGA-II）の探索を行う。"""
# %% Step 1 — 準備｜開発期間の最後の折（学習 day 28〜132、負例 10%、検証 day 133〜146）。軽い族（LR・LightGBM・XGBoost・myFM）だけを使う。
import time

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

from common import SEED, logloss, savefig, show
from common.bench import timed
from common.data import cv_folds
from common.features import load_top, split
from common.models.gbdt import correct, downsample, fit_predict, frame, levels_of
from common.models.linear import NumScaler, design, hash_codes
from common.tuning import counts, run_or_load, suggest

R = 0.1
df = load_top()
df = df[df.day <= 146]
lv = levels_of(df)
f = cv_folds()[-1]
tr = downsample(split(df, *f["train"]), "purchased", R, seed=SEED)
va = split(df, *f["valid"])
y = va.purchased.to_numpy()
req = va.sample(300, random_state=0)


def fit_eval(fam, p, with_time=False):
    """族とパラメーターで学習し、検証の LogLoss（補正後）と、300 件の採点の P99（ms）を返す。"""
    if fam == "lr":
        sc = NumScaler().fit(tr)
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(C=p["C"], max_iter=300).fit(design(tr, sc, hash_codes(tr), 16), tr.purchased)
        q = m.predict_proba(design(va, sc, hash_codes(va), 16))[:, 1]
        t = timed(lambda: m.predict_proba(design(req, sc, hash_codes(req), 16))) if with_time else None
    else:
        q, m = fit_predict(fam, p, frame(tr, lv), tr.purchased.to_numpy(), frame(va, lv), seed=SEED)
        if with_time:
            m.set_params(n_jobs=1)
            Xr = frame(req, lv)
            t = timed(lambda: m.predict_proba(Xr))
        else:
            t = None
    return logloss(y, correct(q, R)), t


# %% Step 2 — 族を探索対象に｜1 つの study の中で、まず族（lr・lgbm・xgb）を選び、族ごとのパラメーターを提案する（30 試行）。
def objective(trial):
    fam = trial.suggest_categorical("family", ["lr", "lgbm", "xgb"])
    s = lambda *a, **k: suggest(trial, fam, *a, **k)
    if fam == "lr":
        p = {"C": s("C", "float", 1e-3, 10.0, log=True)}
    elif fam == "lgbm":
        p = {"n_estimators": s("n_estimators", "int", 100, 800, log=True), "learning_rate": s("learning_rate", "float", 0.01, 0.2, log=True),
             "num_leaves": s("num_leaves", "int", 15, 255, log=True), "min_child_samples": s("min_child_samples", "int", 20, 500, log=True)}
    else:
        p = {"n_estimators": s("n_estimators", "int", 100, 800, log=True), "learning_rate": s("learning_rate", "float", 0.01, 0.2, log=True),
             "max_depth": s("max_depth", "int", 3, 10)}
    return fit_eval(fam, p)[0]


st = run_or_load("11_family", objective, n_trials=30)
t = st.trials_dataframe(attrs=("value", "params", "state"))
t = t[t.state == "COMPLETE"]
g = t.groupby("params_family").value.agg(["count", "min"]).rename(columns={"count": "試行数", "min": "最良の LogLoss"})
show(g, "族を探索対象にした study（30 試行）", digits=5)
print("最良の族:", st.best_params["family"], f"LogLoss {st.best_value:.5f}", counts(st))

# %% Step 3 — 2 つの目的｜LightGBM の木の数・葉の数・学習率を、LogLoss と P99 の両方を小さくするように NSGA-II で探す（30 試行）。
def objective2(trial):
    p = {"n_estimators": trial.suggest_int("n_estimators", 50, 1500, log=True), "num_leaves": trial.suggest_int("num_leaves", 7, 255, log=True),
         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True), "min_child_samples": 50}
    return fit_eval("lgbm", p, with_time=True)


st2 = run_or_load("11_pareto", objective2, n_trials=30, directions=("minimize", "minimize"))
front = sorted(st2.best_trials, key=lambda t: t.values[1])
pf = pd.DataFrame([{**t.params, "LogLoss": t.values[0], "P99（ms）": t.values[1]} for t in front])
show(pf, "パレート解（LogLoss と P99 の両方で、他に負けない構成）", index=False, digits=4)

# %% Step 4 — 規則で 1 つ選ぶ｜P99 が予算（10 ms）以内のパレート解のうち、LogLoss が最小のもの。
ok = pf[pf["P99（ms）"] <= 10.0]
pick = ok.loc[ok.LogLoss.idxmin()]
print("採る構成:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in pick.items()})
all_t = pd.DataFrame([{"LogLoss": t.values[0], "P99": t.values[1]} for t in st2.trials if t.values])
fig, ax = plt.subplots(figsize=(5.6, 4))
ax.scatter(all_t.P99, all_t.LogLoss, s=12, color="lightgray", label="全試行")
ax.plot(pf["P99（ms）"], pf.LogLoss, marker="o", label="パレート解")
ax.axvline(10, color="gray", ls="--", lw=0.8, label="P99 の予算 10 ms")
ax.scatter([pick["P99（ms）"]], [pick.LogLoss], s=80, facecolors="none", edgecolors="red", label="採る構成")
ax.set(xlabel="P99（ms、300 件の採点、1 スレッド）", ylabel="検証の LogLoss")
ax.legend(fontsize=7)
savefig(fig, "11_selection", "pareto")
