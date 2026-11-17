"""第10章：クーポンの Uplift。1 回目の配信（day 42）で学習し、2 回目（day 126）で、誰に送るべきかを当てられるかを比べる。"""
# %% Step 1 — 特徴量｜配信日より前の記録だけで、会員ごとの行動・購入・返品・属性を作る（第3章の時点の約束と同じ）。
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklift.metrics import qini_auc_score, qini_curve

from common import SEED, connect, savefig, show

con = connect()
D = "date_diff('day', DATE '2025-04-07', CAST({} AS DATE))"
X = con.sql(f"""
WITH a AS (SELECT ca.assignment_id, ca.campaign_id, ca.user_id, (ca."group" = 'treatment')::INT AS t, ca.purchased_14d::INT AS y,
                  {D.format('ca.sent_at')} AS d FROM campaign_assignment ca),
v AS (SELECT a.assignment_id, SUM(({D.format('e.occurred_at')} >= a.d - 7)::INT) AS v7, COUNT(*) AS v28
      FROM a JOIN event e ON e.user_id = a.user_id AND e.event_type = 'view' AND e.source <> 'landing'
       AND {D.format('e.occurred_at')} BETWEEN a.d - 28 AND a.d - 1 GROUP BY 1),
o AS (SELECT a.assignment_id, COUNT(*) AS n_buy, AVG(ln(l.unit_price)) AS lp, MAX({D.format('o.ordered_at')}) AS last_buy,
             SUM((r.return_id IS NOT NULL AND {D.format('r.returned_at')} < a.d)::INT) AS n_ret
      FROM a JOIN "order" o ON o.user_id = a.user_id AND {D.format('o.ordered_at')} < a.d
      JOIN order_line l USING (order_id) LEFT JOIN "return" r USING (order_line_id) GROUP BY 1)
SELECT a.*, COALESCE(v.v7, 0) AS v7, COALESCE(v.v28, 0) AS v28, COALESCE(o.n_buy, 0) AS n_buy, COALESCE(o.lp, 0) AS lp,
       COALESCE(a.d - o.last_buy, 999) AS since_buy, COALESCE(o.n_ret * 1.0 / NULLIF(o.n_buy, 0), 0) AS ret_rate,
       a.d - {D.format('u.registered_at')} AS tenure, u.member_rank, u.age_band, u.gender, u.device, u.mail_opt_in::INT AS optin
FROM a LEFT JOIN v USING (assignment_id) LEFT JOIN o USING (assignment_id) JOIN "user" u ON u.user_id = a.user_id
ORDER BY a.assignment_id""").df()
for c in ("member_rank", "age_band", "gender", "device"):
    X[c] = X[c].astype("category").cat.codes
F = ["v7", "v28", "n_buy", "lp", "since_buy", "ret_rate", "tenure", "member_rank", "age_band", "gender", "device", "optin"]
tr, te = X[X.campaign_id == 1], X[X.campaign_id == 2]
truth = connect(truth=True).sql("SELECT assignment_id, p0, p1, uplift FROM truth_campaign").df()
te = te.merge(truth, on="assignment_id")
print(f"学習（1 回目）{len(tr):,} 人・評価（2 回目）{len(te):,} 人、介入群の購入率 {te[te.t == 1].y.mean():.4f}・対照群 {te[te.t == 0].y.mean():.4f}")

# %% Step 2 — 最小実装｜T-learner（介入群と対照群で別のモデルを学び、差を Uplift とする）と、クラス変換（割付確率 0.5 のとき Uplift = 2P(Z=1) − 1）。
# LightGBM は 1 スレッド・決定的な設定にする（交差適合を含めて結果を毎回同じにするため。データが小さいので遅くならない）
gbm = lambda: lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=50, random_state=SEED, verbose=-1, n_jobs=1, deterministic=True)
m1, m0 = gbm().fit(tr[tr.t == 1][F], tr[tr.t == 1].y), gbm().fit(tr[tr.t == 0][F], tr[tr.t == 0].y)
pred = {"T-learner（最小実装）": m1.predict_proba(te[F])[:, 1] - m0.predict_proba(te[F])[:, 1]}
z = (tr.y == tr.t).astype(int)                                # 介入して買った or 介入せず買わなかった
pred["クラス変換（最小実装）"] = 2 * gbm().fit(tr[F], z).predict_proba(te[F])[:, 1] - 1
resp = m1.predict_proba(te[F])[:, 1]                          # 比較：送ったら買いそうな人（反応のモデル）
pred["反応のモデル（買いそうな人）"] = resp

# %% Step 3 — 発展版｜causalml の S・T・X・DR と、scikit-uplift の TwoModels（学習器はどれも同じ LightGBM）。
from causalml.inference.meta import BaseDRRegressor, BaseSClassifier, BaseTClassifier, BaseXClassifier
from sklift.models import TwoModels

half_tr, half_te = np.full(len(tr), 0.5), np.full(len(te), 0.5)   # 割付確率（無作為割付なので 0.5 と分かっている）
for name, M in (("S-learner（causalml）", BaseSClassifier(learner=gbm())), ("T-learner（causalml）", BaseTClassifier(learner=gbm()))):
    M.fit(X=tr[F].to_numpy(), treatment=tr.t.to_numpy(), y=tr.y.to_numpy())
    pred[name] = M.predict(te[F].to_numpy()).ravel()
xl = BaseXClassifier(outcome_learner=gbm(), effect_learner=lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15,
                                                                         random_state=SEED, verbose=-1, n_jobs=1, deterministic=True))
xl.fit(X=tr[F].to_numpy(), treatment=tr.t.to_numpy(), y=tr.y.to_numpy(), p=half_tr)
pred["X-learner（causalml）"] = xl.predict(te[F].to_numpy(), p=half_te).ravel()
dr = BaseDRRegressor(learner=lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15, random_state=SEED, verbose=-1, n_jobs=1, deterministic=True))
dr.fit(X=tr[F].to_numpy(), treatment=tr.t.to_numpy(), y=tr.y.to_numpy(), p=half_tr, seed=SEED)   # 交差適合の分け方を固定
pred["DR-learner（causalml）"] = dr.predict(te[F].to_numpy()).ravel()
tm = TwoModels(estimator_trmnt=gbm(), estimator_ctrl=gbm(), method="vanilla").fit(tr[F], tr.y, tr.t)
pred["TwoModels（scikit-uplift）"] = tm.predict(te[F])

# %% Step 4 — 評価｜Qini（観測した結果だけで測れる）と、真の Uplift との順位相関（答え合わせ）。
rows = []
for name, p in pred.items():
    rows.append((name, qini_auc_score(te.y, p, te.t), spearmanr(p, te.uplift).statistic))
q = pd.DataFrame(rows, columns=["方法", "Qini AUC", "真の Uplift との順位相関"])
show(q, "Uplift の推定（2 回目の配信、合成データ）", index=False)

# %% Step 5 — 金額｜上位 k% に送ったときの期待利益 = Σ{(p1−p0)×1 件の粗利 − p1×クーポン 1 件の値引き}（真の確率で計算）。
aov = con.sql("SELECT SUM(unit_price) / COUNT(DISTINCT order_id) FROM order_line").fetchall()[0][0]
gpr = con.sql("SELECT SUM(unit_price - unit_cost) / SUM(unit_price) FROM order_line").fetchall()[0][0]
M, C = aov * gpr, aov * 0.10
gain = (te.uplift * M - te.p1 * C).to_numpy()                 # 送った場合の、送らない場合に対する利益の増分（1 人あたり）
print(f"1 件の粗利 {M:,.0f} 円・クーポン 1 件の値引き {C:,.0f} 円（注文の平均から計算）")
rows = [("全員に送る", gain.sum()), ("誰にも送らない", 0.0)]
for name in ["反応のモデル（買いそうな人）", "T-learner（最小実装）", "S-learner（causalml）", "X-learner（causalml）"]:
    o = np.argsort(-pred[name])
    rows += [(f"{name}・上位 {k:.0%}", gain[o[: int(len(o) * k)]].sum()) for k in (0.1, 0.3, 0.5)]
o = np.argsort(-te.uplift.to_numpy())
rows += [("真の Uplift の順（上限）・上位 30%", gain[o[: int(len(o) * 0.3)]].sum())]
show(pd.DataFrame(rows, columns=["送り方", "期待利益の増分（円）"]).round(0), "送り方ごとの期待利益（2 回目の配信、真の確率で計算、合成データ）", index=False)

# %% Step 6 — Qini 曲線｜送る人数を増やしたとき、介入で増えた購入の数（累積）がどう増えるか。
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5.6, 4.2))
for name in ["反応のモデル（買いそうな人）", "T-learner（最小実装）", "S-learner（causalml）"]:
    x, yq = qini_curve(te.y, pred[name], te.t)
    ax.plot(x / len(te), yq, label=name, lw=1.2)
ax.plot([0, 1], [0, yq[-1]], color="gray", ls="--", lw=0.8, label="ランダム")
ax.set(xlabel="送った人の割合", ylabel="増えた購入の数（累積、Qini）")
ax.legend(fontsize=7)
savefig(fig, "10_bandit_uplift", "qini")
