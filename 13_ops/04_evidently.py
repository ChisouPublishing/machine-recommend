"""第13章：発展版の監視（Evidently）。03 と同じ基準と比べる期間で、データのずれの報告を作り、ずれた列を数える。"""
# env: ext-ops
# %% Step 1 — データ｜03 と同じ 4 つの列。基準 day 105〜118、比べる期間 day 133〜146 と、価格 1.3 倍を注入した版。
import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from common import OUT
from common.features import FEATURES

df = pd.read_parquet(FEATURES, columns=["day", "log_price", "days_since_release", "u_views_28d", "i_purch_7d"])
cols = ["log_price", "days_since_release", "u_views_28d", "i_purch_7d"]
ref = df[(df.day >= 105) & (df.day <= 118)][cols].sample(50_000, random_state=0)
cur = df[(df.day >= 133) & (df.day <= 146)][cols].sample(50_000, random_state=0)

# %% Step 2 — 報告｜DataDriftPreset（列ごとに検定を選んでずれを判定する）。報告から、ずれた列の数と割合を取り出す。
for name, c in (("開発期間の末", cur), ("価格 1.3 倍を注入", cur.assign(log_price=cur.log_price + np.log(1.3)))):
    snap = Report([DataDriftPreset()]).run(reference_data=ref, current_data=c)
    m = [x for x in snap.dict()["metrics"] if x["metric_name"].startswith("DriftedColumnsCount")][0]
    print(f"{name}：ずれた列 {m['value']['count']:.0f} / {len(cols)}（割合 {m['value']['share']:.2f}）")
    snap.save_html(str(OUT / "13_ops" / f"evidently_{'inj' if '注入' in name else 'dev'}.html"))
