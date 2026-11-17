"""第11章：採用規則（01）とベンチマーク（02）から、最終選定を決めて意思決定ログを書く。結論は表と重みから計算し直して確かめる。"""
# %% Step 1 — 読む｜採用規則（重み・制約・単純さ）、比較の表、最良との差の表。
import json

import numpy as np
import pandas as pd
import yaml

from common import CODE, OUT, show

rule = yaml.safe_load((CODE / "11_selection" / "adoption_rule.yaml").read_text())
tab = pd.read_csv(OUT / "11_selection" / "tables" / "benchmark.csv")
diff = pd.read_csv(OUT / "11_selection" / "tables" / "differences.csv")
FAMILY = {"LR": 0, "LightGBM": 1, "XGBoost": 1, "CatBoost": 1, "FM（myFM）": 2}   # 単純な順（それ以外は DNN = 3）
INTERP = {"高": 1.0, "中": 0.5, "低": 0.0}

# %% Step 2 — 総合点｜制約（P99 ≤ 予算）を満たす候補について、各指標を 0〜1 にそろえ（良いほど 1）、重みを掛けて足す。
def scores(t: pd.DataFrame, w: dict) -> pd.DataFrame:
    t = t.copy()
    mm = lambda s, better_low: (s.max() - s) / (s.max() - s.min()) if better_low else (s - s.min()) / (s.max() - s.min())
    t["s_LogLoss"], t["s_期待粗利"] = mm(t["LogLoss"], True), mm(t["期待粗利（千円）"], False)
    t["s_P99"], t["s_解釈"] = mm(t["P99（ms）"], True), t["解釈"].map(INTERP)
    t["総合点"] = (w["LogLoss"] * t.s_LogLoss + w["期待粗利"] * t.s_期待粗利 + w["P99"] * t.s_P99 + w["解釈のしやすさ"] * t.s_解釈)
    return t.sort_values("総合点", ascending=False).reset_index(drop=True)


budget = rule["constraints"]["ranking_p99_ms"]
ok = tab[tab["P99（ms）"] <= budget]
out = tab[tab["P99（ms）"] > budget]
sc = scores(ok, rule["weights"])
show(sc[["候補", "LogLoss", "期待粗利（千円）", "P99（ms）", "解釈", "総合点"]], f"総合点（P99 ≤ {budget} ms の {len(ok)} 候補）", index=False, digits=4)
print("制約で外した候補:", ", ".join(f"{r.候補}（P99 {r['P99（ms）']:.1f} ms）" for _, r in out.iterrows()) or "なし")

# %% Step 3 — 単純さの規則｜LogLoss が最良と区別できない（差の 95% 区間が 0 を含む）候補のうち、最も単純な族を採る。
best = tab.sort_values("LogLoss").iloc[0]["候補"]
tie = {best} | set(diff[diff["95%下限"] <= 0]["候補"])
tie = [c for c in sc["候補"] if c in tie]                         # 制約を満たすものだけ、総合点の順
adopt = sorted(tie, key=lambda c: (FAMILY.get(c, 3), -sc.set_index("候補").loc[c, "総合点"]))[0] if tie else sc.iloc[0]["候補"]
runner = next(c for c in sc["候補"] if c != adopt)
print(f"LogLoss の最良：{best}。区別できない候補：{', '.join(tie)}")
print(f"採用：{adopt}　次点：{runner}　基準線：LR")

# %% Step 4 — 確かめ｜表と重みだけから、もう一度計算して同じ結論になるか（ログの結論が手で書き換えられていないかの検査）。
again = scores(pd.read_csv(OUT / "11_selection" / "tables" / "benchmark.csv").query("`P99（ms）` <= @budget"), rule["weights"])
assert list(again["候補"]) == list(sc["候補"]), "総合点の順が一致しない"
print("再計算：総合点の順は一致")
dec = {"adopt": adopt, "runner_up": runner, "baseline": "LR", "tie": tie}
(OUT / "11_selection" / "decision.json").write_text(json.dumps(dec, ensure_ascii=False))
a = sc.set_index("候補").loc[adopt]
log = f"""# 意思決定ログ（第11章、本番の候補の選定）

| 項目 | 内容 |
|---|---|
| 採用規則 | 01_adoption_rule.py で比較の前に固定（主の指標 LogLoss、業務の指標 期待粗利、P99 ≤ {budget} ms、単純さの規則、重み {rule['weights']}） |
| 候補 | {len(tab)} 個（第5〜8章の最良の構成）。制約で外した候補：{', '.join(out['候補']) or 'なし'} |
| 比較 | 統一ベンチマーク（学習 day 28〜118・校正 119〜132・評価 133〜146、種 3 つ） |
| 結論 | 採用 {adopt}（LogLoss {a['LogLoss']:.5f}・期待粗利 {a['期待粗利（千円）']:.0f} 千円・P99 {a['P99（ms）']:.1f} ms） |
| 理由 | LogLoss で最良と区別できない候補（{', '.join(tie)}）のうち、最も単純な族 |
| 次点 | {runner}（採用を除いて総合点が最も高い） |
| 保留 | 合成データでの結論。テスト期間で採用・次点・基準線を 1 回だけ確かめる（07） |
| 再選定の条件 | 週次の監視で LogLoss が採用時より 5% 以上悪化、または P99 が予算を超えたとき（第13章） |
"""
(OUT / "11_selection" / "decision_log.md").write_text(log)
