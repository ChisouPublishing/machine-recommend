import os

import numpy as np
import optuna
import pytest

from common import tuning


@pytest.fixture()
def tmp_out(tmp_path, monkeypatch):
    monkeypatch.setattr(tuning, "OUT", tmp_path)
    monkeypatch.delenv("SUISEN_SEARCH", raising=False)
    return tmp_path


def objective(trial):
    x = tuning.suggest(trial, "toy", "x", "float", -3, 3)
    folds = [{"k": k} for k in range(3)]
    # 2 折目で悪い試行を報告させ、枝刈りを発火させる
    return tuning.cv_objective(trial, folds, lambda f, k: (x - 1) ** 2 + (5 if (k >= 1 and abs(x - 1) > 1.5) else 0))


def test_search_then_load_same_best(tmp_out):
    st = tuning.run_or_load("toy", objective, n_trials=40)
    best = st.best_params
    assert abs(tuning.unprefix(best, "toy")["x"] - 1) < 0.5
    c = tuning.counts(st)
    assert c["完了"] + c["枝刈り"] == 40 and c["枝刈り"] > 0
    st2 = tuning.run_or_load("toy", objective, n_trials=40)       # 履歴があるので探索しない
    assert st2.best_params == best and len(st2.trials) == 40


def test_search_flag_redoes(tmp_out, monkeypatch):
    tuning.run_or_load("toy", objective, n_trials=10)
    monkeypatch.setenv("SUISEN_SEARCH", "1")
    st = tuning.run_or_load("toy", objective, n_trials=12)
    assert len(st.trials) == 12


def test_remeasure():
    m, s = tuning.remeasure(lambda seed: np.random.default_rng(seed).normal(1, 0.1))
    assert 0.8 < m < 1.2 and s > 0


def test_remeasure_cached(tmp_out, monkeypatch):
    calls = []
    fn = lambda s: (calls.append(s) or (0.1 * s, 0.5))
    p = {"a": 1, "b": 0.25}
    v1 = tuning.remeasure_cached("toy", p, fn, seeds=(0, 1))
    v2 = tuning.remeasure_cached("toy", p, fn, seeds=(0, 1))     # 同じ構成なので読み込む
    assert v1 == v2 == [(0.0, 0.5), (0.1, 0.5)] and calls == [0, 1]
    tuning.remeasure_cached("toy", {"a": 2, "b": 0.25}, fn, seeds=(0, 1))   # 構成が違えば測り直す
    assert calls == [0, 1, 0, 1]
    monkeypatch.setenv("SUISEN_SEARCH", "1")
    tuning.remeasure_cached("toy", {"a": 2, "b": 0.25}, fn, seeds=(0, 1))   # --search なら測り直す
    assert calls == [0, 1, 0, 1, 0, 1]
