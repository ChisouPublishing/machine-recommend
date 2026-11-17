"""Optuna の探索をそろえて行うための共通処理（第11章の「選定の約束」を実装したもの）。

- 探索の履歴は outputs/optuna/<名前>.db（SQLite）に保存する。既定は保存済みの履歴を読み、
  環境変数 SUISEN_SEARCH=1（run_all.py --search）か、履歴が無いときだけ探索する。
- パラメーター名にはモデル族の接頭辞を付ける（族をまたいで同じ名前が別の意味にならないように）。
- 時系列 CV の折ごとに trial.report し、枝刈りの対象にする。
"""
from __future__ import annotations

import os
import time
from typing import Callable

import numpy as np
import optuna

from .paths import OUT, SEED

optuna.logging.set_verbosity(optuna.logging.WARNING)


def storage_path(name: str):
    p = OUT / "optuna"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{name}.db"


def searching(name: str) -> bool:
    return os.environ.get("SUISEN_SEARCH") == "1" or not storage_path(name).exists()


def get_study(name: str, directions=("minimize",), pruner=None, sampler=None, fresh: bool = False) -> optuna.Study:
    path = storage_path(name)
    if fresh and path.exists():
        path.unlink()
    if sampler is None:
        sampler = (optuna.samplers.TPESampler(seed=SEED) if len(directions) == 1
                   else optuna.samplers.NSGAIISampler(seed=SEED))
    return optuna.create_study(study_name=name, storage=f"sqlite:///{path}", directions=list(directions),
                               sampler=sampler, pruner=pruner or optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
                               load_if_exists=True)


def run_or_load(name: str, objective: Callable, n_trials: int, directions=("minimize",), pruner=None,
                timeout: float | None = None) -> optuna.Study:
    """履歴があれば読み込み、無ければ（または --search なら）探索して保存する。"""
    if searching(name):
        st = get_study(name, directions, pruner, fresh=True)
        t0 = time.time()
        st.optimize(objective, n_trials=n_trials, timeout=timeout, gc_after_trial=True)
        st.set_user_attr("search_seconds", round(time.time() - t0, 1))
        return st
    return get_study(name, directions, pruner)


def counts(st: optuna.Study) -> dict:
    s = [t.state for t in st.trials]
    TS = optuna.trial.TrialState
    return {"完了": s.count(TS.COMPLETE), "枝刈り": s.count(TS.PRUNED), "失敗": s.count(TS.FAIL),
            "探索秒": st.user_attrs.get("search_seconds", float("nan"))}


def suggest(trial: optuna.Trial, family: str, name: str, kind: str, *args, **kw):
    """接頭辞付きのパラメーターを提案する（例：lgbm__num_leaves）。返り値は接頭辞なしの名前で使う。"""
    fn = {"int": trial.suggest_int, "float": trial.suggest_float, "cat": trial.suggest_categorical}[kind]
    return fn(f"{family}__{name}", *args, **kw)


def unprefix(params: dict, family: str) -> dict:
    pre = f"{family}__"
    return {k[len(pre):]: v for k, v in params.items() if k.startswith(pre)}


def cv_objective(trial: optuna.Trial, folds: list, fit_score: Callable[[dict, int], float]) -> float:
    """時系列 CV の折ごとに fit_score(fold, k) を呼び、平均を返す。折ごとに報告して枝刈りに備える。"""
    vals = []
    for k, f in enumerate(folds):
        vals.append(fit_score(f, k))
        trial.report(float(np.mean(vals)), step=k)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(vals))


def remeasure(fn: Callable[[int], float], seeds=(0, 1, 2, 3, 4)) -> tuple[float, float]:
    """最良の構成をシードを変えて測り直す（探索の最良値は楽観的に偏るため）。平均と標準偏差。"""
    v = np.array([fn(s) for s in seeds], float)
    return float(v.mean()), float(v.std(ddof=1))


def remeasure_cached(name: str, params: dict, fn: Callable[[int], tuple], seeds=(0, 1, 2, 3, 4)) -> list:
    """測り直しの結果（種ごとの fn(seed) の値）を outputs/optuna/<名前>_remeasure.json に保存し、既定では読み込む。

    構成（params）と種が保存したものと同じときだけ読み込む。--search（SUISEN_SEARCH=1）のときは測り直す。
    """
    import json
    path = OUT / "optuna" / f"{name}_remeasure.json"
    key = {"params": {k: (round(v, 12) if isinstance(v, float) else v) for k, v in params.items()}, "seeds": list(seeds)}
    if os.environ.get("SUISEN_SEARCH") != "1" and path.exists():
        saved = json.loads(path.read_text())
        if saved["key"] == key:
            return [tuple(v) for v in saved["values"]]
    vals = [tuple(float(x) for x in fn(s)) for s in seeds]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"key": key, "values": vals}))
    return vals
