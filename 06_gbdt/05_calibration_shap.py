"""第6章：校正（Platt・isotonic の最小実装とライブラリ）と、SHAP による解釈。"""
# %% Step 1 — 3 つの期間に分ける｜学習 day 28〜118、校正 day 119〜132、評価 day 133〜146（校正に評価の期間を使わない）。
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from common import OUT, SEED, connect, savefig, show, summary
from common.features import load_top, split
from common.models.calibration import Isotonic, Platt, logit
from common.models.gbdt import GB_COLS, correct, downsample, fit_predict, frame, levels_of

df = load_top()
lv = levels_of(df)
tr, ca, va = split(df, 28, 118), split(df, 119, 132), split(df, 133, 146)
cfg = json.loads((OUT / "06_gbdt" / "best_gbdt.json").read_text())["lgbm"]
d = downsample(tr, "purchased", 0.1, seed=SEED)
q, model = fit_predict("lgbm", cfg, frame(d, lv), d.purchased.to_numpy(), frame(pd.concat([ca, va]), lv), seed=SEED)
p_all = correct(q, 0.1)
p_ca, p_va = p_all[: len(ca)], p_all[len(ca):]
y_ca, y_va = ca.purchased.to_numpy(), va.purchased.to_numpy()

# %% Step 2 — 校正する｜校正の期間で、最小実装（Platt・isotonic）と scikit-learn（ロジスティック回帰・IsotonicRegression）を当てはめる。
pl, iso = Platt().fit(p_ca, y_ca), Isotonic().fit(p_ca, y_ca)
lib_pl = LogisticRegression(C=1e6).fit(logit(p_ca)[:, None], y_ca)
lib_iso = IsotonicRegression(out_of_bounds="clip").fit(p_ca, y_ca)
cand = {"校正なし（間引きの補正のみ）": p_va, "Platt（最小実装）": pl.transform(p_va),
        "Platt（scikit-learn）": lib_pl.predict_proba(logit(p_va)[:, None])[:, 1],
        "isotonic（最小実装）": iso.transform(p_va), "isotonic（scikit-learn）": lib_iso.predict(p_va)}

# %% Step 3 — 評価｜実測のラベルでの指標に加え、真の購入確率との差（区間ごとの |予測 − 真の確率| の平均）を出す。
t = connect(truth=True).sql("SELECT impression_id, p_purchase FROM truth_impression").df()
pt = va[["impression_id"]].merge(t, on="impression_id").p_purchase.to_numpy()


def vs_truth(p):
    b = np.array_split(np.argsort(p), 15)
    return float(np.mean([abs(p[i].mean() - pt[i].mean()) for i in b]))


res = pd.DataFrame({k: {**summary(y_va, p), "真の確率との差": vs_truth(p)} for k, p in cand.items()}).T
show(res.drop(columns="実測率"), "校正の比較（評価 day 133〜146、LightGBM、合成データ）")

# %% Step 4 — SHAP｜評価の期間から 5,000 行を抜き出し、TreeExplainer で各特徴量の寄与（対数オッズ）を計算する。
s = frame(va.sample(5_000, random_state=SEED), lv)
sv = shap.TreeExplainer(model.booster_).shap_values(s)
sv = sv[1] if isinstance(sv, list) else sv
imp = pd.Series(np.abs(sv).mean(0), index=GB_COLS).sort_values(ascending=False)
gain = pd.Series(model.booster_.feature_importance("gain"), index=GB_COLS)
show(pd.DataFrame({"平均 |SHAP|": imp.head(8), "gain の順位": gain.rank(ascending=False).astype(int)[imp.head(8).index]}),
     "SHAP の重要度（上位 8）と gain の順位", digits=4)

# %% Step 5 — 依存の形｜会員のそのカテゴリの閲覧数（u_cat_views_28d）の値ごとに、SHAP の平均を見る（効き方が線形か）。
j = GB_COLS.index("u_cat_views_28d")
b = pd.cut(s["u_cat_views_28d"], [-1, 0, 2, 5, 10, 20, 1e9], labels=["0", "1-2", "3-5", "6-10", "11-20", "21-"])
show(pd.Series(sv[:, j]).groupby(pd.Series(b).reset_index(drop=True), observed=True).mean().rename("SHAP の平均"),
     "u_cat_views_28d の値ごとの寄与", digits=3)
fig, ax = plt.subplots(figsize=(5.6, 3.8))
imp.head(10)[::-1].plot.barh(ax=ax)
ax.set(xlabel="平均 |SHAP|（対数オッズ）")
savefig(fig, "06_gbdt", "shap_importance")
