"""全章のスクリプトを番号順に実行し、標準出力・標準エラー・実行情報を outputs/ に保存する。

使い方:
  python run_all.py                  # 全章（合成データが無ければ先に生成）
  python run_all.py 05_lr 06_gbdt    # 指定した章だけ
  python run_all.py --search         # Optuna の探索をやり直す（既定は保存済みの履歴を読む）
  python run_all.py 06_gbdt/05_calibration_shap.py   # スクリプトを 1 本だけ

スクリプトの先頭近くに `# env: ext-recbole` のように書くと、その別環境（.venv-recbole）で実行する。
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

CODE = Path(__file__).resolve().parent
ENVS = {"main": CODE / ".venv" / "bin" / "python", "ext-recbole": CODE / ".venv-recbole" / "bin" / "python",
        "ext-ops": CODE / ".venv-ops" / "bin" / "python"}
ENV_EXTRA = {"ext-recbole": {"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"}}
CH_RE = re.compile(r"^\d\d_[a-z0-9_]+$")
SC_RE = re.compile(r"^\d\d_[a-z0-9_]+\.py$")


def env_of(script: Path) -> str:
    for line in script.read_text(encoding="utf-8").splitlines()[:15]:
        m = re.match(r"#\s*env:\s*(\S+)", line)
        if m:
            return m.group(1)
    return "main"


def freeze(env: str) -> None:
    py = ENVS[env]
    out = CODE / "outputs" / f"env-{env}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    pv = subprocess.run([str(py), "-c", "import sys,platform;print(sys.version.split()[0], platform.platform())"],
                        capture_output=True, text=True).stdout.strip()
    fr = subprocess.run(["uv", "pip", "freeze", "--python", str(py)], capture_output=True, text=True).stdout
    out.write_text(f"# Python {pv}\n{fr}", encoding="utf-8")


def run_script(script: Path, extra_env: dict) -> dict:
    env_name = env_of(script)
    py = ENVS[env_name]
    rel = script.relative_to(CODE)
    out = CODE / "outputs" / rel.parent
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(CODE), MLFLOW_DISABLE_AGENT_HINT="1", PYTHONHASHSEED="0",
               **ENV_EXTRA.get(env_name, {}), **extra_env)
    t0 = time.time()
    started = datetime.now().isoformat(timespec="seconds")
    p = subprocess.run([str(py), str(rel)], cwd=CODE, env=env, capture_output=True, text=True)
    meta = {"script": str(rel), "env": env_name, "returncode": p.returncode, "seconds": round(time.time() - t0, 1),
            "started": started}
    (out / f"{script.stem}.stdout.txt").write_text(p.stdout, encoding="utf-8")
    (out / f"{script.stem}.stderr.txt").write_text(p.stderr, encoding="utf-8")
    (out / f"{script.stem}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("chapters", nargs="*")
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--regen", action="store_true", help="合成データを作り直す")
    a = ap.parse_args()
    extra = {"SUISEN_SEARCH": "1"} if a.search else {}
    data = CODE / "data" / "generated" / "default"
    if a.regen or not (data / "meta.json").exists():
        print(">>> 合成データを生成します")
        subprocess.run([str(ENVS["main"]), "data/make_data.py"], cwd=CODE, check=True)
    chapters = sorted(d for d in CODE.iterdir() if d.is_dir() and CH_RE.match(d.name))
    singles = [CODE / c for c in a.chapters if c.endswith(".py")]
    if a.chapters:
        chapters = [d for d in chapters if d.name in a.chapters]
    used, failed = set(), []
    jobs = [s for ch in chapters for s in sorted(f for f in ch.iterdir() if SC_RE.match(f.name))] + singles
    for s in jobs:
        if True:
            m = run_script(s, extra)
            used.add(m["env"])
            mark = "OK" if m["returncode"] == 0 else "NG"
            print(f"{mark} {m['script']:45s} {m['env']:12s} {m['seconds']:7.1f} 秒")
            if m["returncode"]:
                failed.append(m["script"])
    for e in sorted(used):
        freeze(e)
    print(f"Python {platform.python_version()}（run_all）、失敗 {len(failed)} 件")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
