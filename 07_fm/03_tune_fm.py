"""第7章：FM を 3 つの実装（最小実装・FuxiCTR・myFM）で、同じ予算（各 12 試行）の Optuna で調整し、測り直して比べる。"""
# %% Step 1 — 準備｜3 折の時系列 CV。学習は負例 10%。FuxiCTR の入力は折ごとに 1 回だけ作って使い回す（語彙は各折の学習期間から）。
import common  # noqa: F401  先に読み込む（LightGBM を PyTorch より先に。付録 A）
import json

import myfm
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder

from common import OUT, SEED, auc, logloss, show
from common.data import cv_folds
from common.features import load_top
from common.models import fuxi
from common.models.gbdt import correct, downsample
from common.models.torch_ctr import FIELDS, FM, Encoder, train
from common.tuning import counts, cv_objective, remeasure_cached, run_or_load, suggest, unprefix

torch.set_num_threads(1)   # 並列の足し合わせの順で結果が揺れないよう、1 スレッドで学習する
R, TRIALS = 0.1, 12
df = load_top()
df = df[df.day <= 146]                                   # 開発期間だけ（テストは読まない）
folds = cv_folds()
cache = {}


def parts(k, seed=SEED):
    f = folds[k]
    a, b = f["train"]
    tr = downsample(df[(df.day >= a) & (df.day <= b)], "purchased", R, seed=seed)
    va = df[(df.day >= f["valid"][0]) & (df.day <= f["valid"][1])]
    return tr, va


def fit_score(impl, p, k, seed=SEED, metric=logloss):
    tr, va = parts(k, seed)
    y = va.purchased.to_numpy()
    if impl == "mine":
        enc = Encoder().fit(tr)
        m = FM(enc.sizes(), len(enc.scaler.mu), k=p["k"])
        q = train(m, enc.transform(tr), tr.purchased.to_numpy(), enc.transform(va), epochs=p["epochs"], lr=p["lr"],
                  weight_decay=p["wd"], seed=seed)
    elif impl == "fuxi":
        key = (k, seed)
        if key not in cache:                                  # 早期終了用は、学習期間の末の 7 日
            cut = tr.day.max() - 6
            cache[key] = fuxi.prepare(f"ch07_tune_{k}_{seed}", tr[tr.day < cut], tr[tr.day >= cut], va)
        q = fuxi.fit_predict("FM", cache[key], {"embedding_dim": p["k"], "learning_rate": p["lr"], "epochs": p["epochs"],
                                                "regularizer": p["wd"]}, seed=seed)
    else:
        oh = OneHotEncoder(handle_unknown="ignore").fit(tr[FIELDS].astype(str))
        enc = Encoder().fit(tr)
        X = lambda d: sparse.hstack([oh.transform(d[FIELDS].astype(str)), sparse.csr_matrix(enc.scaler.transform(d))]).tocsr()
        m = myfm.MyFMClassifier(rank=p["k"], random_seed=seed)
        m.fit(X(tr), tr.purchased.to_numpy(), n_iter=p["n_iter"], n_kept_samples=p["n_iter"] // 2)
        q = m.predict_proba(X(va))
    p_ = correct(q, R)
    return (logloss(y, p_), auc(y, p_)) if metric == "both" else metric(y, p_)


# %% Step 2 — 探索空間｜埋め込みの次元・学習率・エポック・正則化（myFM はベイズ推定なので、次元と標本の数）。
def space(t, impl):
    s = lambda *a, **kw: suggest(t, impl, *a, **kw)
    if impl == "myfm":
        return {"k": s("k", "int", 4, 16, log=True), "n_iter": s("n_iter", "int", 40, 150, log=True)}
    return {"k": s("k", "int", 4, 32, log=True), "lr": s("lr", "float", 1e-3, 3e-2, log=True),
            "epochs": s("epochs", "int", 1, 5), "wd": s("wd", "float", 1e-8, 1e-4, log=True)}


# %% Step 3 — 探索と測り直し｜各 12 試行。最良の構成を、乱数の種 5 つで開発期間の最後の折に当てはめて測り直す（結果は保存し、既定では読む）。
NAMES = {"mine": "FM（最小実装）", "fuxi": "FM（FuxiCTR）", "myfm": "FM（myFM、ベイズ）"}
rows, best = [], {}
for impl in NAMES:
    st = run_or_load(f"07_fm_{impl}", lambda t, impl=impl: cv_objective(t, folds, lambda f, k: fit_score(impl, space(t, impl), k)),
                     n_trials=TRIALS)
    c = counts(st)
    best[impl] = p = unprefix(st.best_params, impl)
    ll, au = zip(*remeasure_cached(f"07_fm_{impl}", p, lambda s: fit_score(impl, p, 2, seed=s, metric="both")))
    rows.append((NAMES[impl], c["完了"], c["枝刈り"], st.best_value, np.mean(ll), np.std(ll, ddof=1), np.mean(au), np.std(au, ddof=1)))
    print(f"[time] {impl} の探索 {c['探索秒']} 秒")
show(pd.DataFrame(rows, columns=["実装", "完了", "枝刈り", "CV の LogLoss（最良）", "測り直し LogLoss", "標準偏差", "AUC", "AUC 標準偏差"]),
     "FM の 3 つの実装（測り直しは検証 day 133〜146・種 5 つ、合成データ）", index=False, digits=5)
for impl, p in best.items():
    print(NAMES[impl], {k: (round(v, 5) if isinstance(v, float) else v) for k, v in p.items()})
(OUT / "07_fm").mkdir(parents=True, exist_ok=True)
(OUT / "07_fm" / "best_fm.json").write_text(json.dumps(best))
