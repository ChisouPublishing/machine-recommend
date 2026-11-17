"""第6章：Optuna で 3 つのライブラリを同じ予算（20 試行）で調整し、最良の構成を乱数の種を変えて測り直す。"""
# %% Step 1 — 準備｜3 折の時系列 CV。学習は負例を 10% に間引いて速くし、予測は補正して元の水準に戻す（03 で確かめた方法）。
import json

import numpy as np
import pandas as pd

from common import OUT, SEED, auc, logloss, show
from common.data import cv_folds
from common.features import load_top
from common.models.gbdt import correct, downsample, fit_predict, frame, levels_of
from common.tuning import counts, cv_objective, remeasure_cached, run_or_load, suggest, unprefix

df = load_top()
df = df[df.day <= 146]                                  # 開発期間だけ（テストは読まない）
lv = levels_of(df)
folds = cv_folds()
R, TRIALS = 0.1, 20


def fit_score(lib, params, fold, seed=SEED, metric=logloss):
    a, b = fold["train"]
    tr = downsample(df[(df.day >= a) & (df.day <= b)], "purchased", R, seed=seed)
    va = df[(df.day >= fold["valid"][0]) & (df.day <= fold["valid"][1])]
    q, _ = fit_predict(lib, params, frame(tr, lv), tr.purchased.to_numpy(), frame(va, lv), seed=seed)
    y, p = va.purchased.to_numpy(), correct(q, R)
    return (logloss(y, p), auc(y, p)) if metric == "both" else metric(y, p)


# %% Step 2 — 探索空間｜ライブラリごとに、木の数・学習率・木の大きさ・葉の最小量・抽出の割合・L2 を探す（名前に接頭辞を付ける）。
def space(trial, lib):
    s = lambda *a, **k: suggest(trial, lib, *a, **k)
    common = {"learning_rate": s("learning_rate", "float", 0.01, 0.2, log=True),
              "subsample": s("subsample", "float", 0.5, 1.0), "colsample_bytree": s("colsample_bytree", "float", 0.5, 1.0)}
    if lib == "lgbm":
        return {**common, "n_estimators": s("n_estimators", "int", 100, 800, log=True), "subsample_freq": 1,
                "num_leaves": s("num_leaves", "int", 15, 255, log=True),
                "min_child_samples": s("min_child_samples", "int", 20, 500, log=True),
                "reg_lambda": s("reg_lambda", "float", 1e-3, 10.0, log=True)}
    if lib == "xgb":
        return {**common, "n_estimators": s("n_estimators", "int", 100, 800, log=True),
                "max_depth": s("max_depth", "int", 3, 10), "min_child_weight": s("min_child_weight", "float", 1, 100, log=True),
                "reg_lambda": s("reg_lambda", "float", 1e-3, 10.0, log=True)}
    return {"iterations": s("n_estimators", "int", 100, 800, log=True), "learning_rate": common["learning_rate"],
            "depth": s("depth", "int", 4, 8), "l2_leaf_reg": s("l2_leaf_reg", "float", 1.0, 30.0, log=True),
            "bootstrap_type": "Bernoulli", "subsample": common["subsample"], "rsm": common["colsample_bytree"]}


# %% Step 3 — 探索｜ライブラリごとに 20 試行（保存した履歴があれば読む）。完了・枝刈りの数と探索の秒数を記録する。
best, rows = {}, []
for lib in ["lgbm", "xgb", "cat"]:
    st = run_or_load(f"06_gbdt_{lib}", lambda t, lib=lib: cv_objective(t, folds, lambda f, k: fit_score(lib, space(t, lib), f)),
                     n_trials=TRIALS)
    c = counts(st)
    fixed = {"subsample_freq": 1} if lib == "lgbm" else ({"bootstrap_type": "Bernoulli"} if lib == "cat" else {})
    p = unprefix(st.best_params, lib)
    if lib == "cat":
        p = {"iterations": p.pop("n_estimators"), "rsm": p.pop("colsample_bytree"), **p}
    best[lib] = {**p, **fixed}
    rows.append((lib, c["完了"], c["枝刈り"], st.best_value))
    print(f"[time] {lib} の探索 {c['探索秒']} 秒")
show(pd.DataFrame(rows, columns=["ライブラリ", "完了", "枝刈り", "CV の LogLoss（探索の最良）"]), "探索の結果（各 20 試行）", index=False, digits=5)

# %% Step 4 — 測り直し｜最良の構成を、乱数の種 5 つ（間引きと学習の乱数）で開発期間の最後の折に当てはめ、平均と標準偏差を出す（結果は保存し、既定では読む）。
rem = []
for lib, p in best.items():
    ll, au = zip(*remeasure_cached(f"06_gbdt_{lib}", p, lambda s: fit_score(lib, p, folds[-1], seed=s, metric="both")))
    rem.append((lib, np.mean(ll), np.std(ll, ddof=1), np.mean(au), np.std(au, ddof=1)))
show(pd.DataFrame(rem, columns=["ライブラリ", "LogLoss 平均", "LogLoss 標準偏差", "AUC 平均", "AUC 標準偏差"]),
     "最良の構成の測り直し（検証 day 133〜146、種 5 つ、合成データ）", index=False, digits=5)
for lib, p in best.items():
    print(lib, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.items()})
(OUT / "06_gbdt").mkdir(parents=True, exist_ok=True)
(OUT / "06_gbdt" / "best_gbdt.json").write_text(json.dumps(best))
