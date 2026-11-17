"""第3章：トップのおすすめの特徴量の表を作り（common/features.py）、中身を確かめる。"""
# %% Step 1 — 作る｜特徴量の表を作り直して保存する（約 10 秒）。以降の章はこの表を読む。
import pandas as pd

from common import PERIODS, show
from common.features import CAT_COLS, NUM_COLS, SEQ_COLS, load_top

df = load_top(rebuild=True)

# %% Step 2 — 期間ごとの件数とラベルの率｜開発・テストの行数と、クリック率・購入率。
per = {k: df[(df.day >= a) & (df.day <= b)] for k, (a, b) in PERIODS.items() if k in ("dev", "gap", "test")}
show(pd.DataFrame([(k, len(v), v.clicked.mean(), v.purchased.mean()) for k, v in per.items()],
                  columns=["期間", "行数", "クリック率", "購入率"]), "期間ごとの件数とラベル", index=False)

# %% Step 3 — 特徴量の一覧｜分類ごとの列数と、数値の列の要約（欠損の割合・中央値・99% 点）。
print(f"カテゴリの列 {len(CAT_COLS)}、数値の列 {len(NUM_COLS)}、系列の列 {len(SEQ_COLS)}")
d = per["dev"][NUM_COLS]
show(pd.DataFrame({"欠損": d.isna().mean(), "中央値": d.median(), "99%点": d.quantile(0.99)}).round(3),
     "数値の特徴量（開発期間）", digits=3)
print("閲覧の系列が空の割合:", round((per["dev"].hist_items == "").mean(), 4))
