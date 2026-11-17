"""第11章：統一ベンチマーク。どの候補も、同じ期間・同じ間引き・同じ校正・同じ指標で学習・評価する。

期間：学習 day 28〜118、校正 day 119〜132、評価 day 133〜146（いずれも開発期間。テストは使わない）。
間引き：LR 以外は負例を 10% に間引いて学習し、予測を補正する（第6章）。
校正：校正の期間の予測で Platt scaling を当てはめ、評価の期間に使う（第6章）。
速さ：1 回の推薦で 300 件の候補を採点する時間を 200 回測り、99% 点（P99）をミリ秒で出す（1 スレッド）。
"""
# %% Step 1 — 候補の一覧｜各章で調整した最良の構成を読み、候補ごとに「学習して予測を返す」関数を用意する。
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch

from common.models.calibration import Platt
from common.models.gbdt import correct, downsample, fit_predict as gb_fit, frame, levels_of
from common.models.linear import NumScaler, design, hash_codes
from common.paths import OUT

R = 0.1
PERIOD = {"train": (28, 118), "cal": (119, 132), "eval": (133, 146)}
CANDS = {  # 名前: (族, 実装, 解釈のしやすさ)
    "LR": ("lr", "scikit-learn", "高"), "LightGBM": ("gbdt", "ライブラリ", "中"), "XGBoost": ("gbdt", "ライブラリ", "中"),
    "CatBoost": ("gbdt", "ライブラリ", "中"), "FM（myFM）": ("fm", "ライブラリ", "低"),
    "DeepFM（最小実装）": ("dnn", "自作", "低"), "DeepFM（FuxiCTR）": ("dnn", "ライブラリ", "低"),
    "DCN-V2（FuxiCTR）": ("dnn", "ライブラリ", "低"), "DIN（最小実装）": ("dnn", "自作", "低"), "DIN（FuxiCTR）": ("dnn", "ライブラリ", "低")}


def best(chapter: str, name: str):
    return json.loads((OUT / chapter / name).read_text())


def timed(fn, n_req=200):
    """fn() を n_req 回呼び、1 回の時間の 99% 点（ミリ秒）を返す。"""
    fn()
    ts = []
    for _ in range(n_req):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    return float(np.percentile(ts, 99))


# %% Step 2 — 学習と予測｜候補の族ごとに学習し、校正と評価の期間の予測（補正済み）と、300 件の採点の P99 を返す。
def between(df, ab):
    return df[(df.day >= ab[0]) & (df.day <= ab[1])]


def rest_of(df, periods=PERIOD):
    """校正と評価の期間の行（この順）。間の期間（第11章の最終評価では day 147〜153）は含めない。"""
    return pd.concat([between(df, periods["cal"]), between(df, periods["eval"])])


def run(name: str, df: pd.DataFrame, seed: int, periods=PERIOD):
    tr_all = between(df, periods["train"])
    rest = rest_of(df, periods)
    req = rest.sample(300, random_state=0)                              # 1 回の推薦で採点する 300 件
    torch.set_num_threads(1)
    if name == "LR":
        cfg = best("05_lr", "best_lr.json")
        from sklearn.linear_model import LogisticRegression
        sc = NumScaler().fit(tr_all)
        m = LogisticRegression(C=cfg["C"], max_iter=300).fit(design(tr_all, sc, hash_codes(tr_all), cfg["bits"]), tr_all.purchased)
        p = m.predict_proba(design(rest, sc, hash_codes(rest), cfg["bits"]))[:, 1]
        p99 = timed(lambda: m.predict_proba(design(req, sc, hash_codes(req), cfg["bits"])))
        return p, p99
    tr = downsample(tr_all, "purchased", R, seed=seed)
    if name in ("LightGBM", "XGBoost", "CatBoost"):
        lib = {"LightGBM": "lgbm", "XGBoost": "xgb", "CatBoost": "cat"}[name]
        lv = levels_of(df)
        q, m = gb_fit(lib, best("06_gbdt", "best_gbdt.json")[lib], frame(tr, lv), tr.purchased.to_numpy(), frame(rest, lv), seed=seed)
        if lib == "cat":
            from catboost import Pool
            from common.models.gbdt import GB_CATS
            Xr = frame(req, lv).astype({c: str for c in GB_CATS})
            cats = [Xr.columns.get_loc(c) for c in GB_CATS]
            p99 = timed(lambda: m.predict_proba(Pool(Xr, cat_features=cats), thread_count=1))
        else:
            m.set_params(n_jobs=1)
            Xr = frame(req, lv)
            p99 = timed(lambda: m.predict_proba(Xr))
        return correct(q, R), p99
    if name == "FM（myFM）":
        import myfm
        from scipy import sparse
        from sklearn.preprocessing import OneHotEncoder
        from common.models.torch_ctr import FIELDS
        cfg = best("07_fm", "best_fm.json")["myfm"]
        oh = OneHotEncoder(handle_unknown="ignore").fit(tr[FIELDS].astype(str))
        sc = NumScaler().fit(tr)
        X = lambda d: sparse.hstack([oh.transform(d[FIELDS].astype(str)), sparse.csr_matrix(sc.transform(d))]).tocsr()
        m = myfm.MyFMClassifier(rank=cfg["k"], random_seed=seed)
        m.fit(X(tr), tr.purchased.to_numpy(), n_iter=cfg["n_iter"], n_kept_samples=cfg["n_iter"] // 2)
        p99 = timed(lambda: m.predict_proba(X(req)))
        return correct(m.predict_proba(X(rest)), R), p99
    cfgs = best("08_dnn", "best_dnn.json")
    if name.endswith("（最小実装）"):
        from common.features import NUM_COLS
        from common.models.torch_ctr import DIN, DeepFM, Encoder, predict, train
        cfg = cfgs["mine:" + name.split("（")[0]]
        M = {"DeepFM": DeepFM, "DIN": DIN}[name.split("（")[0]]
        enc = Encoder().fit(tr)
        m = M(enc.sizes(), len(NUM_COLS), k=cfg["k"], hidden=(cfg["h1"], cfg["h1"] // 2), dropout=cfg["dropout"])
        torch.set_num_threads(1)
        q = train(m, enc.transform(tr), tr.purchased.to_numpy(), enc.transform(rest), epochs=cfg["epochs"], lr=cfg["lr"],
                  weight_decay=cfg["wd"], seed=seed)
        torch.set_num_threads(1)
        xr = enc.transform(req)
        p99 = timed(lambda: predict(m, xr))
        return correct(q, R), p99
    from common.models import fuxi
    model_name = {"DeepFM（FuxiCTR）": "DeepFM", "DCN-V2（FuxiCTR）": "DCNv2", "DIN（FuxiCTR）": "DIN"}[name]
    cfg = cfgs["fuxi:" + model_name]
    seq = model_name == "DIN"
    cut = periods["train"][1] - 6
    data = fuxi.prepare(f"bench_{model_name}_{seed}", tr[tr.day < cut], tr[tr.day >= cut], rest, with_seq=seq)
    hid = [cfg["h1"], cfg["h1"] // 2]
    o = {"embedding_dim": cfg["k"], "learning_rate": cfg["lr"], "epochs": cfg["epochs"], "net_dropout": cfg["dropout"],
         "embedding_regularizer": cfg["wd"]}
    o.update({"DeepFM": {"hidden_units": hid}, "DIN": {"dnn_hidden_units": hid},
              "DCNv2": {"parallel_dnn_hidden_units": hid, "stacked_dnn_hidden_units": hid}}[model_name])
    torch.set_num_threads(1)
    q, model, batch = fuxi.fit_predict(model_name, data, o, seed=seed, return_model=True, sample_batch=300)
    torch.set_num_threads(1)
    model.eval()
    with torch.no_grad():
        p99 = timed(lambda: model.forward(batch))
    return correct(q, R), p99


# %% Step 3 — 校正｜校正の期間で Platt を当てはめ、評価の期間の確率に使う。
def calibrate(p_all: np.ndarray, rest: pd.DataFrame, periods=PERIOD):
    cal = ((rest.day >= periods["cal"][0]) & (rest.day <= periods["cal"][1])).to_numpy()
    pl = Platt().fit(p_all[cal], rest.purchased.to_numpy()[cal])
    return pl.transform(p_all[~cal])
