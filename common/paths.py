import os
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("SUISEN_DATA", CODE / "data" / "generated" / "default"))
OUT = CODE / "outputs"
SEED = 20261116


def out_dir(chapter: str, sub: str = "") -> Path:
    p = OUT / chapter / sub if sub else OUT / chapter
    p.mkdir(parents=True, exist_ok=True)
    return p
