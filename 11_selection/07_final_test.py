"""第11章：最終評価。採用・次点・基準線（LR）の 3 つだけを、テスト期間（day 154〜181）で 1 回だけ評価する。"""
# %% Step 1 — 1 回だけ｜結果が保存済みなら、テスト期間を読まずにそれを表示する（やり直すのは SUISEN_FINAL=1 のときだけ）。
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

from common import OUT, auc, ece, logloss, show
from common.bench import calibrate, rest_of, run
from common.features import load_top

res_path, log_path = OUT / "11_selection" / "final_test.json", OUT / "11_selection" / "test_reads.log"
dec = json.loads((OUT / "11_selection" / "decision.json").read_text())
names = [dec["adopt"], dec["runner_up"], dec["baseline"]]
P = {"train": (28, 132), "cal": (133, 146), "eval": (154, 181)}     # 開発期間の全体で学習・校正し、間の 1 週をあけてテスト

if os.environ.get("SUISEN_FINAL") == "1" or not res_path.exists():
    # %% Step 2 — 学習と評価｜統一ベンチマークと同じ手順（間引き・校正）で、乱数の種 3 つの予測の平均をテスト期間で評価する。
    df = load_top()
    rest = rest_of(df, P)
    y = rest[rest.day >= P["eval"][0]].purchased.to_numpy()
    preds = {n: np.mean([calibrate(run(n, df, s, P)[0], rest, P) for s in range(3)], axis=0) for n in names}
    with open(log_path, "a") as fh:
        fh.write(f"{datetime.now().isoformat(timespec='seconds')} test day {P['eval']} {names}\n")
    # %% Step 3 — 不確かさ｜テストの行のブートストラップ（200 回）で、指標の 95% 区間と、基準線（LR）との差の区間・Holm 補正の p 値。
    rng = np.random.default_rng(0)
    B = [rng.integers(0, len(y), len(y)) for _ in range(200)]
    out = {}
    for n in names:
        ll = np.array([logloss(y[i], preds[n][i]) for i in B])
        au = np.array([auc(y[i], preds[n][i]) for i in B])
        d = ll - np.array([logloss(y[i], preds["LR"][i]) for i in B])
        out[n] = {"LogLoss": logloss(y, preds[n]), "LogLoss 95%": list(np.quantile(ll, [0.025, 0.975])), "AUC": auc(y, preds[n]),
                  "AUC 95%": list(np.quantile(au, [0.025, 0.975])), "ECE": ece(y, preds[n]),
                  "差（対 LR）": float(d.mean()), "差 95%": list(np.quantile(d, [0.025, 0.975])),
                  "p": float(min(1, 2 * min((d <= 0).mean(), (d >= 0).mean())))}
    ps = sorted([(out[n]["p"], n) for n in names if n != "LR"])
    for k, (p, n) in enumerate(ps):
        out[n]["Holm"] = float(min(1, max(p * (len(ps) - j) for j, (p, _) in enumerate(ps[: k + 1]))))
    out["LR"]["Holm"] = float("nan")
    res_path.write_text(json.dumps(out, ensure_ascii=False))
# %% Step 4 — 記録｜MLflow（outputs/11_selection/mlflow.db）に、3 つの候補の最終評価を 1 回だけ残す（保存した結果から。テストは読まない）。
out = json.loads(res_path.read_text())
mark = OUT / "11_selection" / "mlflow_logged"
if not mark.exists():
    import contextlib
    import io

    import mlflow
    mlflow.set_tracking_uri(f"sqlite:///{OUT / '11_selection' / 'mlflow.db'}")
    mlflow.set_experiment("final_test")
    for n in names:
        with mlflow.start_run(run_name=n), contextlib.redirect_stdout(io.StringIO()):
            mlflow.log_params({"candidate": n, "test_days": str(P["eval"])})
            mlflow.log_metrics({"logloss": out[n]["LogLoss"], "auc": out[n]["AUC"], "ece": out[n]["ECE"]})
    mark.write_text("logged")

# %% Step 5 — 表示｜保存した結果と、テスト期間を読んだ回数（記録の行数）。
rows = [(n, v["LogLoss"], f"{v['LogLoss 95%'][0]:.5f}〜{v['LogLoss 95%'][1]:.5f}", v["AUC"], v["ECE"], v["差（対 LR）"],
         f"{v['差 95%'][0]:.5f}〜{v['差 95%'][1]:.5f}", v["Holm"]) for n, v in out.items()]
show(pd.DataFrame(rows, columns=["候補", "LogLoss", "95% 区間", "AUC", "ECE", "差（対 LR）", "差の 95% 区間", "Holm 補正の p"]),
     "最終評価（テスト day 154〜181、種 3 つの平均、合成データ）", index=False, digits=5)
print("テスト期間を読んだ回数:", len(log_path.read_text().splitlines()))
