"""データの読み込みと期間の分割。

期間（26 週、day 0 = 2025-04-07）:
  助走      day   0〜27  特徴量の履歴だけに使う（学習の行にしない）
  開発      day  28〜146  学習と検証（時系列 CV はこの中だけ）
  間隔      day 147〜153  本番への反映の遅れを模した 1 週（使わない）
  テスト    day 154〜181  最終評価で 1 回だけ使う
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from .paths import DATA

START = pd.Timestamp("2025-04-07")
PERIODS = {"warmup": (0, 27), "dev": (28, 146), "gap": (147, 153), "test": (154, 181)}
TRUTH = ("truth_impression", "truth_order_line", "truth_campaign", "truth_user")


def day_of(ts) -> np.ndarray:
    return ((pd.to_datetime(ts) - START) // pd.Timedelta(days=1)).to_numpy()


def periods() -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """各期間の開始時刻と終了時刻（終了は翌日 0 時、半開区間）。"""
    return {k: (START + pd.Timedelta(days=a), START + pd.Timedelta(days=b + 1)) for k, (a, b) in PERIODS.items()}


def cv_folds(n: int = 3, weeks: int = 2) -> list[dict]:
    """開発期間の中の拡大窓の時系列 CV。最後の n 個の 2 週間を順に検証に使う。"""
    a, b = PERIODS["dev"]
    folds = []
    for k in range(n, 0, -1):
        v_end = b - (k - 1) * weeks * 7
        v_start = v_end - weeks * 7 + 1
        folds.append({"train": (a, v_start - 1), "valid": (v_start, v_end)})
    return folds


def connect(truth: bool = False, data_dir=None) -> duckdb.DuckDBPyConnection:
    """全テーブルをビューとして持つ DuckDB の接続。truth=True のときだけ答え合わせ用の表も見せる。"""
    d = data_dir or DATA
    con = duckdb.connect()
    for f in sorted(d.glob("*.parquet")):
        if f.stem in TRUTH and not truth:
            continue
        con.execute(f'CREATE VIEW "{f.stem}" AS SELECT * FROM read_parquet(\'{f}\')')
    return con


def load(name: str, columns=None, truth: bool = False, data_dir=None) -> pd.DataFrame:
    if name in TRUTH and not truth:
        raise PermissionError(f"{name} は答え合わせ用です。truth=True を付けたスクリプトだけが読めます")
    return pd.read_parquet((data_dir or DATA) / f"{name}.parquet", columns=columns)
