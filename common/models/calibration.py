"""第6章：確率の校正の最小実装（Platt scaling と、PAV による isotonic regression）。"""
# %% Step 1 — Platt scaling｜予測の対数オッズ z に、a·z + b のロジスティック回帰を当てはめる（2 つの数だけ学ぶ）。
import numpy as np


def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


class Platt:
    def fit(self, p, y, iters=50):
        z, a, b = logit(p), 1.0, 0.0
        for _ in range(iters):                                  # ニュートン法（2 変数）
            q = 1 / (1 + np.exp(-(a * z + b)))
            g = np.array([((q - y) * z).sum(), (q - y).sum()])
            w = q * (1 - q)
            H = np.array([[(w * z * z).sum(), (w * z).sum()], [(w * z).sum(), w.sum()]])
            a, b = np.array([a, b]) - np.linalg.solve(H + 1e-9 * np.eye(2), g)
        self.a, self.b = a, b
        return self

    def transform(self, p):
        return 1 / (1 + np.exp(-(self.a * logit(p) + self.b)))


# %% Step 2 — isotonic regression｜予測の順に並べ、実測の率が単調に増えるよう、逆転した隣の区間を平均でまとめる（PAV）。
class Isotonic:
    def fit(self, p, y):
        o = np.argsort(p, kind="stable")
        xs, ys = p[o], y[o].astype(float)
        val, wt, right = [], [], []                             # 区間ごとの平均・件数・右端の予測
        for x, v in zip(xs, ys):
            val.append(v); wt.append(1.0); right.append(x)
            while len(val) > 1 and val[-2] > val[-1]:           # 単調性が崩れたら、直前の区間とまとめる
                w = wt[-2] + wt[-1]
                val[-2] = (val[-2] * wt[-2] + val[-1] * wt[-1]) / w
                wt[-2], right[-2] = w, right[-1]
                val.pop(); wt.pop(); right.pop()
        self.right, self.val = np.array(right), np.array(val)
        return self

    def transform(self, p):
        k = np.clip(np.searchsorted(self.right, p, side="left"), 0, len(self.val) - 1)
        return self.val[k]
