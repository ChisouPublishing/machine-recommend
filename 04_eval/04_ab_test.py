"""第4章：A/B テストの必要サンプルサイズ。トップのおすすめの CTR と購入率で、検出したい差ごとに必要な表示数と日数を出す。"""
# %% Step 1 — 準備｜開発期間のトップの表示数（1 日あたり）と、CTR・購入率の水準を読む。
import numpy as np
import pandas as pd
from scipy.stats import norm

from common import show
from common.features import load_top, split

df = split(load_top(), 28, 146)
per_day = len(df) / df.day.nunique()
base = {"CTR": df.clicked.mean(), "購入率": df.purchased.mean()}

# %% Step 2 — 必要な数｜2 つの割合の差の検定（両側 5%、検出力 80%）で、群ごとに必要な表示数を計算する。
def n_per_arm(p, lift, alpha=0.05, power=0.8):
    p2 = p * (1 + lift)
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return z ** 2 * (p * (1 - p) + p2 * (1 - p2)) / (p2 - p) ** 2

# %% Step 3 — 日数に直す｜トップの表示を半分ずつ 2 群に分けたときに、何日かかるかを並べる。
rows = []
for m, p in base.items():
    for lift in (0.02, 0.05, 0.10):
        n = n_per_arm(p, lift)
        rows.append((m, round(p, 4), f"+{lift:.0%}", int(np.ceil(n)), round(n / (per_day / 2), 1)))
print(f"トップの表示：1 日あたり {per_day:,.0f} 件（開発期間）")
show(pd.DataFrame(rows, columns=["指標", "水準", "検出したい差（相対）", "1 群に必要な表示数", "必要な日数"]),
     "A/B テストの必要サンプルサイズ（両側 5%、検出力 80%）", index=False)
