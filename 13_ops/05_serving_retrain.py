"""第13章：配信（関数の呼び出し → FastAPI）の速さと、再学習の頻度（1 回だけ・毎週）による精度の違い。"""
# %% Step 1 — モデル｜第6章の LightGBM（最良の構成）を day 28〜118 で学習する。配信では 300 件の候補を 1 回で採点する。
import json
import time

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from common import OUT, SEED, logloss, show
from common.features import load_top, split
from common.models.gbdt import correct, downsample, fit_predict, frame, levels_of

df = load_top()
lv = levels_of(df)
cfg = json.loads((OUT / "06_gbdt" / "best_gbdt.json").read_text())["lgbm"]
tr = downsample(split(df, 28, 118), "purchased", 0.1, seed=SEED)
_, model = fit_predict("lgbm", cfg, frame(tr, lv), tr.purchased.to_numpy(), frame(tr.head(1), lv), seed=SEED)
model.set_params(n_jobs=1)
pool = split(df, 133, 146).sample(300, random_state=0)
X = frame(pool, lv)

# %% Step 2 — 最小実装：関数｜採点の関数を直接呼ぶ。
score = lambda Xc: correct(model.predict_proba(Xc)[:, 1], 0.1)


def p99(fn, n=200):
    fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1000)
    return np.percentile(ts, 99)


# %% Step 3 — 発展版：FastAPI｜候補の行（特徴量）を JSON で受け取り、上位 6 件の位置と確率を返す API。TestClient で同じ 300 件を送る。
app = FastAPI()


class Req(BaseModel):
    rows: list[dict]


@app.post("/rank")
def rank(req: Req):
    p = score(frame(pd.DataFrame(req.rows), lv))
    top = np.argsort(-p)[:6]
    return {"top": top.tolist(), "p": p[top].round(5).tolist()}


client = TestClient(app)
body = {"rows": json.loads(pool[X.columns.tolist()].to_json(orient="records"))}
r = client.post("/rank", json=body).json()
same = r["top"] == np.argsort(-score(X))[:6].tolist()
print("API の上位 6 件が関数と一致:", same)
print(f"[time] P99：関数 {p99(lambda: score(X)):.1f} ms・FastAPI（JSON の変換を含む）{p99(lambda: client.post('/rank', json=body)):.1f} ms")

# %% Step 4 — 再学習の頻度｜day 105 で 1 回だけ学習したモデルと、毎週（7 日ごと）学習し直すモデルを、day 105〜146 の週ごとの LogLoss で比べる。
rows = []
once = None
for w0 in range(105, 147, 7):
    va = split(df, w0, w0 + 6)
    if once is None:
        t0 = downsample(split(df, 28, 104), "purchased", 0.1, seed=SEED)
        once, _ = fit_predict("lgbm", cfg, frame(t0, lv), t0.purchased.to_numpy(), frame(split(df, 105, 146), lv), seed=SEED)
        once = pd.Series(correct(once, 0.1), index=split(df, 105, 146).impression_id)
    tw = downsample(split(df, 28, w0 - 1), "purchased", 0.1, seed=SEED)
    q, _ = fit_predict("lgbm", cfg, frame(tw, lv), tw.purchased.to_numpy(), frame(va, lv), seed=SEED)
    rows.append((f"day {w0}〜{w0 + 6}", logloss(va.purchased, once.loc[va.impression_id].to_numpy()), logloss(va.purchased, correct(q, 0.1))))
t = pd.DataFrame(rows, columns=["週", "1 回だけ（day 104 まで）", "毎週学習し直す"])
t["差"] = t["毎週学習し直す"] - t["1 回だけ（day 104 まで）"]
show(t, "再学習の頻度と LogLoss（開発期間、合成データ）", index=False, digits=5)
