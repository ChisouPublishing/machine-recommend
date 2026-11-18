"""第8章 8.6：クリックと購入を 1 つのモデルで学ぶ多タスクの最小実装（PyTorch）。Shared-Bottom・ESMM・MMoE と、2 つの損失の学習ループ。

どのモデルも、表示ごとに 2 つの値を返す：pCTR（表示 → クリック）と pCVR（クリック → 購入）。表示 → 購入（CTCVR）は pCTR × pCVR。
"""
# %% Step 1 — 共通の入力｜第7章の Encoder の出力（カテゴリの番号・標準化した数値）を、列ごとの埋め込みにしてつなぐ。
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from common.models.torch_ctr import init_


def mlp(d_in, hidden, dropout=0.1, out=None):
    layers = []
    for h in hidden:
        layers += [nn.Linear(d_in, h), nn.ReLU(), nn.Dropout(dropout)]
        d_in = h
    if out is not None:
        layers.append(nn.Linear(d_in, out))
    return nn.Sequential(*layers)


class Embed(nn.Module):
    def __init__(self, sizes, n_num, k=8):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(s, k) for s in sizes])
        self.dim = len(sizes) * k + n_num

    def forward(self, x):
        E = torch.cat([e(x["cat"][:, i]) for i, e in enumerate(self.emb)], 1)
        return torch.cat([E, x["num"]], 1)


# %% Step 2 — 3 つの構造｜(1) Shared-Bottom：下の MLP を共有し、上に 2 つの塔。(2) ESMM：埋め込みだけを共有し、CTR と CVR の塔は別。
#               (3) MMoE：複数の専門家（MLP）を、タスクごとの門（softmax の重み）で混ぜて各塔に渡す。
class SharedBottom(nn.Module):
    def __init__(self, sizes, n_num, k=8, bottom=(128,), tower=(64,), dropout=0.1):
        super().__init__()
        self.inp = Embed(sizes, n_num, k)
        self.bottom = mlp(self.inp.dim, bottom, dropout)
        self.towers = nn.ModuleList([mlp(bottom[-1], tower, dropout, out=1) for _ in range(2)])
        init_(self)

    def forward(self, x):
        h = self.bottom(self.inp(x))
        return self.towers[0](h).squeeze(1), self.towers[1](h).squeeze(1)       # (CTR の logit, CVR の logit)


class ESMM(nn.Module):
    def __init__(self, sizes, n_num, k=8, hidden=(128, 64), dropout=0.1):
        super().__init__()
        self.inp = Embed(sizes, n_num, k)                                       # 埋め込みは共有（クリックの多い行で CVR の側も学べる）
        self.ctr = mlp(self.inp.dim, hidden, dropout, out=1)
        self.cvr = mlp(self.inp.dim, hidden, dropout, out=1)
        init_(self)

    def forward(self, x):
        z = self.inp(x)
        return self.ctr(z).squeeze(1), self.cvr(z).squeeze(1)


class MMoE(nn.Module):
    def __init__(self, sizes, n_num, k=8, n_experts=4, expert=(128,), tower=(64,), dropout=0.1):
        super().__init__()
        self.inp = Embed(sizes, n_num, k)
        self.experts = nn.ModuleList([mlp(self.inp.dim, expert, dropout) for _ in range(n_experts)])
        self.gates = nn.ModuleList([nn.Linear(self.inp.dim, n_experts) for _ in range(2)])
        self.towers = nn.ModuleList([mlp(expert[-1], tower, dropout, out=1) for _ in range(2)])
        init_(self)

    def forward(self, x):
        z = self.inp(x)
        H = torch.stack([e(z) for e in self.experts], 1)                        # (行, 専門家, 次元)
        out = []
        for g, t in zip(self.gates, self.towers):
            w = torch.softmax(g(z), -1).unsqueeze(-1)                           # タスクごとに、専門家の混ぜ方を変える
            out.append(t((w * H).sum(1)).squeeze(1))
        return out[0], out[1]


# %% Step 3 — 2 つの損失｜"click_only"：CTR は全表示、CVR はクリックした表示だけで学ぶ（従来の作り方）。
#               "esmm"：CTR と CTCVR（= pCTR × pCVR）をどちらも全表示で学ぶ。CVR に直接の正解は与えない。w は CTCVR の損失の重み。
def _bce_prob(p, y):
    p = p.clamp(1e-7, 1 - 1e-7)
    return -(y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean()


def loss_of(mode, lc, lv, click, buy, w=1.0):
    bce = nn.functional.binary_cross_entropy_with_logits
    l_ctr = bce(lc, click)
    if mode == "click_only":
        m = click > 0
        l_cvr = bce(lv[m], buy[m]) if m.any() else lc.sum() * 0
        return l_ctr + w * l_cvr
    if mode == "esmm":
        return l_ctr + w * _bce_prob(torch.sigmoid(lc) * torch.sigmoid(lv), buy)
    raise ValueError(mode)


def reinit(model, seed):
    torch.manual_seed(seed)
    for m in model.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    init_(model)


def train(model, xtr, click, buy, mode, epochs=2, lr=2e-3, w=1.0, batch=4096, seed=0):
    """click・buy：表示ごとの 0/1。学習が済んだら、モデルを返す（予測は predict）。"""
    reinit(model, seed)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    c = torch.from_numpy(np.asarray(click, np.float32))
    b = torch.from_numpy(np.asarray(buy, np.float32))
    n = len(c)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        for s in range(0, n, batch):
            i = perm[s:s + batch]
            opt.zero_grad()
            lc, lv = model({k: v[i] for k, v in xtr.items()})
            loss_of(mode, lc, lv, c[i], b[i], w).backward()
            opt.step()
    return model


@torch.no_grad()
def predict(model, x, batch=65536):
    """(pCTR, pCVR) を返す。pCTCVR は 2 つの積。"""
    model.eval()
    n = len(x["cat"])
    pc, pv = [], []
    for s in range(0, n, batch):
        lc, lv = model({k: v[s:s + batch] for k, v in x.items()})
        pc.append(torch.sigmoid(lc)); pv.append(torch.sigmoid(lv))
    return torch.cat(pc).numpy().astype(np.float64), torch.cat(pv).numpy().astype(np.float64)
