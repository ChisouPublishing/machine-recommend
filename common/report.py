"""結果の表示と保存（本文に載る標準出力の書式をそろえる）。"""
from __future__ import annotations

import time
import unicodedata
from contextlib import contextmanager

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .paths import out_dir  # noqa: E402

import logging  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)   # 無い字体の候補の警告を出さない
plt.rcParams["font.family"] = ["Hiragino Sans", "Noto Sans JP", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
pd.set_option("display.unicode.east_asian_width", True)
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)


def _width(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)


def show(df: pd.DataFrame | pd.Series, title: str | None = None, digits: int = 4, index: bool = True,
         left: tuple = ()) -> None:
    """表を表示する。left に挙げた列は左寄せにする（木の形の KPI など）。"""
    if title:
        print(f"■ {title}")
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.round(digits)
    if left:
        df = df.copy()
        for c in left:
            vals = df[c].astype(str)
            w = max(_width(v) for v in list(vals) + [str(c)])
            df[c] = [v + " " * (w - _width(v)) for v in vals]
        df.columns = [str(c) + " " * (max(_width(v) for v in df[c].astype(str)) - _width(str(c))) if c in left else c
                      for c in df.columns]
    print(df.to_string(index=index))
    print()


def save_table(df: pd.DataFrame, chapter: str, name: str, digits: int = 4) -> None:
    df.round(digits).to_csv(out_dir(chapter, "tables") / f"{name}.csv", index=False)


def savefig(fig, chapter: str, name: str) -> None:
    fig.savefig(out_dir(chapter, "figs") / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


@contextmanager
def timer(label: str):
    t = time.perf_counter()
    yield
    print(f"[time] {label} {time.perf_counter() - t:.1f} 秒")
