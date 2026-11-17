import numpy as np
from sklearn.metrics import roc_auc_score

from common import metrics as M
from common.data import cv_folds, PERIODS


def test_auc_logloss_known():
    y = np.array([0, 0, 1, 1]); p = np.array([0.1, 0.4, 0.35, 0.8])
    assert abs(M.auc(y, p) - 0.75) < 1e-12
    assert abs(M.logloss(y, p) - (-(np.log(0.9) + np.log(0.6) + np.log(0.35) + np.log(0.8)) / 4)) < 1e-12


def test_ece_perfect_and_biased():
    r = np.random.default_rng(0); p = r.random(200_000); y = (r.random(200_000) < p).astype(int)
    assert M.ece(y, p) < 0.01
    assert abs(M.ece(y, np.clip(p + 0.1, 0, 1)) - 0.1) < 0.02


def test_ndcg_recall():
    assert abs(M.ndcg_at_k([[1, 0, 0]], 3) - 1.0) < 1e-12
    assert abs(M.ndcg_at_k([[0, 1]], 2) - (1 / np.log2(3))) < 1e-12
    assert M.recall_at_k({1: [5, 6, 7]}, {1: {6, 9}}, k=2) == 0.5


def test_bootstrap_contains_point():
    r = np.random.default_rng(1); y = r.integers(0, 2, 2000); p = np.clip(y * 0.3 + r.random(2000) * 0.7, 0, 1)
    lo, hi = M.bootstrap_ci(y, p, M.auc, n=100)
    assert lo < roc_auc_score(y, p) < hi


def test_cv_folds_inside_dev_and_ordered():
    a, b = PERIODS["dev"]
    f = cv_folds()
    assert [x["valid"] for x in f] == [(105, 118), (119, 132), (133, 146)]
    for x in f:
        assert x["train"][0] == a and x["train"][1] < x["valid"][0] and x["valid"][1] <= b
