"""第8章：DeepFM・DIN（最小実装と FuxiCTR）と DCN-V2（FuxiCTR）を、同じ予算（各 10 試行）の Optuna で調整し、測り直す。"""
# %% Step 1 — 準備｜3 折の時系列 CV（学習は負例 10%）。FuxiCTR の入力は折ごと・系列の有無ごとに 1 回だけ作って使い回す。
import common  # noqa: F401  先に読み込む（LightGBM を PyTorch より先に。付録 A）
import json

import numpy as np
import pandas as pd
import torch

from common import OUT, SEED, auc, logloss, show
from common.data import cv_folds
from common.features import NUM_COLS, load_top
from common.models import fuxi
from common.models.gbdt import correct, downsample
from common.models.torch_ctr import DIN, DeepFM, Encoder, train
from common.tuning import counts, cv_objective, remeasure_cached, run_or_load, suggest, unprefix

torch.set_num_threads(1)   # 並列の足し合わせの順で結果が揺れないよう、1 スレッドで学習する
R, TRIALS, SEEDS = 0.1, 10, 3
df = load_top()
df = df[df.day <= 146]                                   # 開発期間だけ（テストは読まない）
folds = cv_folds()
cache = {}


def parts(k):
    f = folds[k]
    tr = downsample(df[(df.day >= f["train"][0]) & (df.day <= f["train"][1])], "purchased", R, seed=SEED)
    return tr, df[(df.day >= f["valid"][0]) & (df.day <= f["valid"][1])]


def fit_score(cand, p, k, seed=SEED, metric=logloss):
    impl, model = cand.split(":")
    tr, va = parts(k)
    if impl == "mine":
        enc = Encoder().fit(tr)
        M = {"DeepFM": DeepFM, "DIN": DIN}[model]
        m = M(enc.sizes(), len(NUM_COLS), k=p["k"], hidden=(p["h1"], p["h1"] // 2), dropout=p["dropout"])
        q = train(m, enc.transform(tr), tr.purchased.to_numpy(), enc.transform(va), epochs=p["epochs"], lr=p["lr"],
                  weight_decay=p["wd"], seed=seed)
    else:
        seq = model in ("DIN", "DIEN")
        key = (k, seq)
        if key not in cache:
            cut = tr.day.max() - 6
            cache[key] = fuxi.prepare(f"ch08_tune_{k}_{int(seq)}", tr[tr.day < cut], tr[tr.day >= cut], va, with_seq=seq)
        hid = [p["h1"], p["h1"] // 2]
        o = {"embedding_dim": p["k"], "learning_rate": p["lr"], "epochs": p["epochs"], "net_dropout": p["dropout"],
             "embedding_regularizer": p["wd"]}
        o.update({"DeepFM": {"hidden_units": hid}, "DIN": {"dnn_hidden_units": hid},
                  "DCNv2": {"parallel_dnn_hidden_units": hid, "stacked_dnn_hidden_units": hid}}[model])
        q = fuxi.fit_predict(model, cache[key], o, seed=seed)
    y, p_ = va.purchased.to_numpy(), correct(q, R)
    return (logloss(y, p_), auc(y, p_)) if metric == "both" else metric(y, p_)


# %% Step 2 — 探索空間｜埋め込みの次元・1 層目の幅（2 層目はその半分）・ドロップアウト・学習率・エポック・正則化。
def space(t, cand):
    s = lambda *a, **kw: suggest(t, cand.replace(":", "_"), *a, **kw)
    return {"k": s("k", "int", 4, 32, log=True), "h1": s("h1", "cat", [64, 128, 256]),
            "dropout": s("dropout", "float", 0.0, 0.3), "lr": s("lr", "float", 5e-4, 1e-2, log=True),
            "epochs": s("epochs", "int", 1, 4), "wd": s("wd", "float", 1e-8, 1e-4, log=True)}


# %% Step 3 — 探索と測り直し｜各 10 試行。最良の構成を、乱数の種 3 つ（初期値と並べ替え）で開発期間の最後の折に当てはめる（結果は保存し、既定では読む）。
CANDS = {"mine:DeepFM": "DeepFM（最小実装）", "fuxi:DeepFM": "DeepFM（FuxiCTR）", "fuxi:DCNv2": "DCN-V2（FuxiCTR）",
         "mine:DIN": "DIN（最小実装）", "fuxi:DIN": "DIN（FuxiCTR）"}
rows, best = [], {}
for cand, name in CANDS.items():
    tag = cand.replace(":", "_")
    st = run_or_load(f"08_dnn_{tag}", lambda t, c=cand: cv_objective(t, folds, lambda f, k: fit_score(c, space(t, c), k)),
                     n_trials=TRIALS)
    c = counts(st)
    best[cand] = p = unprefix(st.best_params, tag)
    ll, au = zip(*remeasure_cached(f"08_dnn_{tag}", p, lambda s: fit_score(cand, p, 2, seed=s, metric="both"), seeds=range(SEEDS)))
    rows.append((name, c["完了"], c["枝刈り"], st.best_value, np.mean(ll), np.std(ll, ddof=1), np.mean(au), np.std(au, ddof=1)))
    print(f"[time] {cand} の探索 {c['探索秒']} 秒")
show(pd.DataFrame(rows, columns=["モデル", "完了", "枝刈り", "CV の LogLoss（最良）", "測り直し LogLoss", "標準偏差", "AUC", "AUC 標準偏差"]),
     f"深層学習の 5 候補（測り直しは検証 day 133〜146・種 {SEEDS} つ、合成データ）", index=False, digits=5)
for cand, p in best.items():
    print(CANDS[cand], {k: (round(v, 5) if isinstance(v, float) else v) for k, v in p.items()})
(OUT / "08_dnn").mkdir(parents=True, exist_ok=True)
(OUT / "08_dnn" / "best_dnn.json").write_text(json.dumps(best))
