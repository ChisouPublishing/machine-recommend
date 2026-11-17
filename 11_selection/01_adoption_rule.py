"""第11章：採用規則を、比較の前に固定する（一度書いたら書き換えない。日時を残す）。"""
# %% Step 1 — 規則｜業務の物差し・制約・単純さ・総合点の重みを決める。値は業務で決めるもので、ここでは本書の EC の想定。
from datetime import datetime

import yaml

from common import CODE

path = CODE / "11_selection" / "adoption_rule.yaml"
rule = yaml.safe_load(path.read_text())
if not rule.get("fixed_at"):
    rule = {
        "primary_metric": "LogLoss（校正後、評価 day 133〜146）",
        "business_metric": "返品控除後の期待粗利の順に上位 10% の表示を選んだときの、実際の返品控除後の粗利",
        "constraints": {"ranking_p99_ms": 10.0, "note": "300 件の候補の採点、1 スレッド"},
        "simplicity": "主の指標の差の 95% 区間が 0 を含むなら、単純な族を採る（LR < GBDT < FM < DNN）",
        "weights": {"LogLoss": 0.4, "期待粗利": 0.3, "P99": 0.2, "解釈のしやすさ": 0.1},
        "final_test": "採用・次点・基準線（LR）の 3 つだけを、テスト期間で 1 回だけ評価する",
        "fixed_at": datetime.now().isoformat(timespec="seconds"),
    }
    path.write_text(yaml.safe_dump(rule, allow_unicode=True, sort_keys=False))
    print("[time] 採用規則をいま固定した")
else:
    print("[time] 採用規則は固定済み（書き換えない）")

# %% Step 2 — 表示｜固定した規則（日時を除く）を表示する。日時は本文に載せない（実行の記録にだけ残す）。
for k, v in rule.items():
    if k == "fixed_at":
        print(f"[time] 固定した日時 {v}")
    else:
        print(f"{k}: {v}")
