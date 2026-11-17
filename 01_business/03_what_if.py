"""第1章：KPI ツリーで「どこを何% 改善すると粗利がいくら動くか」を見積もる（感度分析）。"""
# %% Step 1 — 準備｜第1章の 2 つの表（KPI ツリーと推薦面ごとの成果）を読み込む。
import pandas as pd

from common import OUT, show

surf = pd.read_csv(OUT / "01_business" / "tables" / "surface.csv")
gp = surf["返品控除後の粗利"].sum()

# %% Step 2 — 改善の仮定｜推薦面ごとに、CTR が 10% 上がる・返品率が 2 ポイント下がる場合を考える。
# 単純化の仮定：クリックが増えても、クリックあたりの購入と客単価は変わらない
rows = []
for _, r in surf.dropna(subset=["表示数"]).iterrows():
    margin_per_line = r["返品控除後の粗利"] / r["明細数"]
    rows.append((r["推薦面"], "CTR +10%", r["返品控除後の粗利"] * 0.10))
    gross_per_line = margin_per_line / (1 - r["返品率"]) if r["返品率"] < 1 else 0
    rows.append((r["推薦面"], "返品率 −2pt", r["明細数"] * 0.02 * (gross_per_line + 1_000)))
w = pd.DataFrame(rows, columns=["推薦面", "改善", "粗利の増分（円）"])
w["全体に対する%"] = (w["粗利の増分（円）"] / gp * 100).round(1)
w["粗利の増分（円）"] = w["粗利の増分（円）"].round().astype("int64")

# %% Step 3 — 結果｜増分の大きい順に並べる。どの推薦面のどの指標に効くと事業への寄与が大きいかが分かる。
show(w.sort_values("粗利の増分（円）", ascending=False), "改善の見積もり（開発期間、合成データ）", index=False, digits=1)
