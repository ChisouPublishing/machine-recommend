"""評価指標（scikit-learn と同じ定義を、小さな関数で揃えて使う）。"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


def auc(y, p) -> float:
    return float(roc_auc_score(y, p))


def logloss(y, p) -> float:
    return float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]))


def pr_auc(y, p) -> float:
    return float(average_precision_score(y, p))


def ece(y, p, bins: int = 15) -> float:
    """期待キャリブレーション誤差（予測確率の等頻度の区間ごとに |平均予測 − 実測率| を件数で重み付け）。"""
    y, p = np.asarray(y, float), np.asarray(p, float)
    order = np.argsort(p, kind="stable")
    err = 0.0
    for idx in np.array_split(order, bins):
        if len(idx):
            err += len(idx) * abs(p[idx].mean() - y[idx].mean())
    return float(err / len(p))


def ndcg_at_k(rel_lists, k: int = 10) -> float:
    """rel_lists：ユーザーごとの、推薦順に並べた関連度（0/1）の配列のリスト。"""
    disc = 1 / np.log2(np.arange(2, k + 2))
    vals = []
    for rel in rel_lists:
        rel = np.asarray(rel, float)
        dcg = (rel[:k] * disc[: len(rel[:k])]).sum()
        ideal = np.sort(rel)[::-1][:k]
        idcg = (ideal * disc[: len(ideal)]).sum()
        if idcg > 0:
            vals.append(dcg / idcg)
    return float(np.mean(vals)) if vals else 0.0


def recall_at_k(recommended: dict, truth: dict, k: int = 10) -> float:
    """recommended・truth：ユーザー → 商品の並び・集合。正解のあるユーザーの平均。"""
    vals = [len(set(recommended.get(u, [])[:k]) & set(t)) / len(t) for u, t in truth.items() if len(t)]
    return float(np.mean(vals)) if vals else 0.0


def bootstrap_ci(y, p, fn, n: int = 200, seed: int = 0, alpha: float = 0.05):
    """指標 fn のブートストラップ信頼区間（行を復元抽出）。"""
    r = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        i = r.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(fn(y[i], p[i]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def summary(y, p) -> dict:
    return {"AUC": auc(y, p), "LogLoss": logloss(y, p), "PR-AUC": pr_auc(y, p), "ECE": ece(y, p),
            "平均予測": float(np.mean(p)), "実測率": float(np.mean(y))}
