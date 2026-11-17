"""第3章：特徴量の時点の検査を、本番の特徴量の表と、わざとリークを入れた表（テスト）の両方で行う。"""
# %% Step 1 — 本番の表を検査｜保存した特徴量の表を、表示の日・セッションの開始より前の記録だけで数え直す。
import re
import subprocess
import sys

from common import CODE, connect
from common.features import load_top
from common.leakage import audit_sequence, audit_user_views

df, con = load_top(), connect()
print("u_views_28d の食い違い:", len(audit_user_views(df, con)), "件 / 2000 件")
print("u_views_7d の食い違い:", len(audit_user_views(df, con, col="u_views_7d", window=7)), "件 / 2000 件")
print("hist_items の食い違い:", len(audit_sequence(df, con)), "件 / 500 セッション")

# %% Step 2 — 検査が働くかを確かめる｜tests/test_leakage.py は、未来の記録を混ぜた表で検査が失敗することを確かめる。
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_leakage.py"], cwd=CODE, capture_output=True, text=True)
for line in r.stdout.splitlines():
    if line.startswith("tests/") or "passed" in line or "failed" in line:
        print("テスト:", re.sub(r" in [0-9.]+s", "", line).strip("= "))
