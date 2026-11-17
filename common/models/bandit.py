"""第10章：文脈付きバンディットの最小実装（ε-greedy・LinUCB・線形 Thompson Sampling）。腕ごとに別の線形モデルを持つ。"""
# %% Step 1 — ε-greedy｜確率 ε で一様に選び、残りは腕ごとの平均の報酬が最大の腕を選ぶ（文脈を使わない）。
import numpy as np


class EpsGreedy:
    def __init__(self, n_arms, eps=0.1, seed=0):
        self.n, self.s, self.eps, self.r = np.zeros(n_arms), np.zeros(n_arms), eps, np.random.default_rng(seed)

    def select(self, x):
        if self.r.random() < self.eps:
            return int(self.r.integers(len(self.n)))
        return int(np.argmax(self.s / np.maximum(self.n, 1) + 1e-9 * self.r.random(len(self.n))))

    def update(self, a, x, reward):
        self.n[a] += 1
        self.s[a] += reward


# %% Step 2 — LinUCB｜腕ごとに報酬を文脈の線形関数で推定し、推定値 ＋ α × 不確かさ（信頼区間の幅）が最大の腕を選ぶ。
class LinUCB:
    def __init__(self, n_arms, dim, alpha=1.0):
        self.A = np.stack([np.eye(dim)] * n_arms)          # 腕ごとの Σ x xᵀ + I
        self.b = np.zeros((n_arms, dim))                   # 腕ごとの Σ r x
        self.alpha = alpha

    def select(self, x):
        Ainv = np.linalg.inv(self.A)
        theta = np.einsum("aij,aj->ai", Ainv, self.b)
        ucb = theta @ x + self.alpha * np.sqrt(np.einsum("i,aij,j->a", x, Ainv, x))
        return int(np.argmax(ucb))

    def update(self, a, x, reward):
        self.A[a] += np.outer(x, x)
        self.b[a] += reward * x


# %% Step 3 — 線形 Thompson Sampling｜腕ごとの係数の事後分布（正規分布）から 1 つ引き、その係数での推定値が最大の腕を選ぶ。
class LinTS(LinUCB):
    def __init__(self, n_arms, dim, v=0.5, seed=0):
        super().__init__(n_arms, dim)
        self.v, self.r = v, np.random.default_rng(seed)

    def select(self, x):
        Ainv = np.linalg.inv(self.A)
        theta = np.einsum("aij,aj->ai", Ainv, self.b)
        draws = np.array([self.r.multivariate_normal(theta[a], self.v ** 2 * Ainv[a]) for a in range(len(theta))])
        return int(np.argmax(draws @ x))
