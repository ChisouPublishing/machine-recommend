"""第11章の 04 から呼ぶ：環境変数 SUISEN_DATA のデータで、LR・LightGBM・DIN（最小実装）を同じ手順で 1 回ずつ評価する。"""
# %% Step 1 — 3 つの候補｜統一ベンチマークと同じ期間・間引き・校正で、乱数の種 0 だけ評価する。
import numpy as np

from common import DATA, auc, logloss
from common.bench import PERIOD, calibrate, rest_of, run
from common.features import load_top

df = load_top()
df = df[df.day <= 146]
rest = rest_of(df)
y = rest[rest.day >= PERIOD["eval"][0]].purchased.to_numpy()
print(f"■ データ：{DATA.name}")
for name in ("LR", "LightGBM", "DIN（最小実装）"):
    p_all, _ = run(name, df, 0)
    p = calibrate(p_all, rest)
    print(f"{name:16s} LogLoss {logloss(y, p):.5f}  AUC {auc(y, p):.4f}")
