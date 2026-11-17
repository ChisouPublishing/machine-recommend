"""第13章：モデルの管理（ファイルの台帳 → MLflow Model Registry）と、監視（PSI・KS の最小実装）。"""
# %% Step 1 — 最小実装の台帳｜モデルのファイルのハッシュ・学習の期間・指標・状態（本番・待機）を JSON の台帳に残し、切り戻せるようにする。
import contextlib
import hashlib
import io
import json
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from common import OUT, SEED, logloss, show
from common.features import NUM_COLS, load_top, split
from common.models.gbdt import correct, downsample, frame, levels_of

reg = OUT / "13_ops" / "registry"
reg.mkdir(parents=True, exist_ok=True)
df = load_top()
lv = levels_of(df)
cfg = json.loads((OUT / "06_gbdt" / "best_gbdt.json").read_text())["lgbm"]
ledger = []


def train_and_register(version, a, b):
    tr = downsample(split(df, a, b), "purchased", 0.1, seed=SEED)
    m = lgb.LGBMClassifier(objective="binary", random_state=SEED, verbose=-1, deterministic=True, force_row_wise=True, **cfg)
    m.fit(frame(tr, lv), tr.purchased)
    blob = pickle.dumps(m.booster_.model_to_string())
    (reg / f"v{version}.pkl").write_bytes(blob)
    va = split(df, b + 1, b + 14)
    ll = logloss(va.purchased, correct(m.predict_proba(frame(va, lv))[:, 1], 0.1))
    ledger.append({"version": version, "sha256": hashlib.sha256(blob).hexdigest()[:12], "train_days": f"{a}-{b}",
                   "valid_logloss": round(ll, 5), "stage": "待機"})
    return m


for v, (a, b) in enumerate([(28, 104), (28, 118), (28, 132)], 1):
    train_and_register(v, a, b)
ledger[-1]["stage"] = "本番"
(reg / "ledger.json").write_text(json.dumps(ledger, ensure_ascii=False, indent=1))
show(pd.DataFrame(ledger), "最小実装の台帳（3 つの版。最新を本番に）", index=False)

# %% Step 2 — 切り戻し｜本番の版に問題が出たら、台帳の状態を書き換えて前の版を本番に戻す。ファイルのハッシュで中身を確かめる。
for r in ledger:
    r["stage"] = "本番" if r["version"] == 2 else ("停止" if r["version"] == 3 else r["stage"])
ok = hashlib.sha256((reg / "v2.pkl").read_bytes()).hexdigest()[:12] == ledger[1]["sha256"]
print("切り戻し後の本番:", [r["version"] for r in ledger if r["stage"] == "本番"], "ハッシュの一致:", ok)

# %% Step 3 — 発展版：MLflow Model Registry｜同じ 3 つの版を登録し、別名（alias）champion で本番を指す。切り戻しは別名を付け替えるだけ。
import mlflow
import mlflow.lightgbm
from mlflow import MlflowClient

mlflow.set_tracking_uri(f"sqlite:///{OUT / '13_ops' / 'mlflow.db'}")
client = MlflowClient()
if mlflow.get_experiment_by_name("registry") is None:                          # モデルのファイルは outputs/13_ops/mlartifacts に置く
    mlflow.create_experiment("registry", artifact_location=(OUT / "13_ops" / "mlartifacts").as_uri())
mlflow.set_experiment("registry")
try:
    client.delete_registered_model("suisen_top_cvr")
except Exception:
    pass
for r in ledger:
    booster = lgb.Booster(model_str=pickle.loads((reg / f"v{r['version']}.pkl").read_bytes()))
    with mlflow.start_run(run_name=f"v{r['version']}"), contextlib.redirect_stdout(io.StringIO()):   # 登録の案内の表示を出さない
        mlflow.log_metric("valid_logloss", r["valid_logloss"])
        mlflow.lightgbm.log_model(booster, name="model", registered_model_name="suisen_top_cvr")
client.set_registered_model_alias("suisen_top_cvr", "champion", "3")
client.set_registered_model_alias("suisen_top_cvr", "champion", "2")         # 切り戻し
print("MLflow の champion:", client.get_model_version_by_alias("suisen_top_cvr", "champion").version,
      "／登録した版の数:", len(client.search_model_versions("name='suisen_top_cvr'")))


# %% Step 4 — 監視の最小実装｜PSI（区間ごとの割合のずれ）と KS 検定で、基準の期間（day 105〜118）と比べた分布のずれを測る。
def psi(a, b, bins=10):
    e = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    pa = np.histogram(a, e)[0] / len(a) + 1e-6
    pb = np.histogram(np.clip(b, e[0], e[-1]), e)[0] / len(b) + 1e-6
    return float(((pb - pa) * np.log(pb / pa)).sum())


ref = split(df, 105, 118)
cur_dev = split(df, 133, 146)
shifted = cur_dev.assign(log_price=cur_dev.log_price + np.log(1.3))          # 注入：値上げ（価格 1.3 倍）を模す
rows = []
for name, cur in (("開発期間の末（day 133〜146）", cur_dev), ("同じ期間に価格 1.3 倍を注入", shifted)):
    for c in ["log_price", "days_since_release", "u_views_28d", "i_purch_7d"]:
        rows.append((name, c, psi(ref[c].to_numpy(), cur[c].to_numpy()), ks_2samp(ref[c], cur[c]).statistic))
t = pd.DataFrame(rows, columns=["比べる期間", "特徴量", "PSI", "KS 統計量"])
t["警告（PSI ≥ 0.2）"] = np.where(t.PSI >= 0.2, "あり", "")
show(t, "分布のずれ（基準：day 105〜118）", index=False, digits=4)
