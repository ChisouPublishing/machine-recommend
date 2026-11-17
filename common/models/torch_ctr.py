"""第7・8章：埋め込みを使う CTR・CVR モデルの最小実装（PyTorch）。FM・FFM・Wide & Deep・DeepFM・DIN と、共通の学習ループ。"""
# %% Step 1 — 入力の形｜カテゴリの列は語彙で番号に（学習に無い値は 0）、数値は標準化、閲覧の系列は直近 20 件の商品の番号に直す。
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from common.features import NUM_COLS
from common.models.linear import NumScaler

LONG_LEN = 100
FIELDS = ["user_id", "item_id", "brand_id", "category_id", "major", "gender", "age_band", "region", "member_rank", "device",
          "body_size", "item_gender", "season", "price_tier", "position", "hour", "weekday"]
SEQ_LEN = 20


class Encoder:
    def fit(self, df: pd.DataFrame, min_count: int = 1):
        self.vocab = {}
        for c in FIELDS:
            vc = df[c].astype(str).value_counts()
            self.vocab[c] = {v: i + 1 for i, v in enumerate(vc[vc >= min_count].index)}   # 0 は「未知」
        self.scaler = NumScaler().fit(df)
        return self

    def sizes(self):
        return [len(self.vocab[c]) + 1 for c in FIELDS]

    def transform(self, df: pd.DataFrame) -> dict:
        cat = np.column_stack([df[c].astype(str).map(self.vocab[c]).fillna(0).to_numpy(np.int64) for c in FIELDS])
        seqs = {}
        spec = [("seq", "hist_items", "item_id", SEQ_LEN), ("seqc", "hist_cats", "category_id", SEQ_LEN)]
        if "hist_items_long" in df:                                          # SIM 用の長い系列（第8章）
            spec += [("lseq", "hist_items_long", "item_id", LONG_LEN), ("lseqc", "hist_cats_long", "category_id", LONG_LEN)]
        for name, col, voc, L in spec:
            seq = np.zeros((len(df), L), dtype=np.int64)                    # 語彙の番号。古い順に並べ、右に詰める
            v = self.vocab[voc]
            for r, h in enumerate(df[col].to_numpy()):
                ids = [v.get(x, 0) for x in h.split("^")[-L:]] if h else []
                if ids:
                    seq[r, -len(ids):] = ids
            seqs[name] = torch.from_numpy(seq)
        return {"cat": torch.from_numpy(cat), "num": torch.from_numpy(self.scaler.transform(df).astype(np.float32)), **seqs}


# %% Step 2 — FM｜1 次の項と、すべての列の組の埋め込みの内積の和。Σ_{i<j}⟨v_i, v_j⟩ = ½{(Σv)² − Σv²} で、列数の 2 乗でなく 1 乗の計算で済む。
def init_(module):
    """埋め込みは小さな値（標準偏差 0.01）で、1 次の項は 0 で始める（大きく始めると交互作用の項が発散する）。"""
    for m in module.modules():
        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.01) if m.embedding_dim > 1 else nn.init.zeros_(m.weight)


class FM(nn.Module):
    def __init__(self, sizes, n_num, k=8):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(s, k) for s in sizes])
        self.lin = nn.ModuleList([nn.Embedding(s, 1) for s in sizes])
        self.num_v = nn.Parameter(torch.randn(n_num, k) * 0.01)             # 数値の列も、値 × 埋め込みとして交互作用に入れる
        self.num_w = nn.Linear(n_num, 1)
        init_(self)

    def vectors(self, x):
        v = [e(x["cat"][:, i]) for i, e in enumerate(self.emb)]
        return torch.cat([torch.stack(v, 1), x["num"].unsqueeze(2) * self.num_v], 1)   # (行, 列, k)

    def forward(self, x):
        V = self.vectors(x)
        inter = 0.5 * ((V.sum(1) ** 2) - (V ** 2).sum(1)).sum(1)
        first = sum(l(x["cat"][:, i]).squeeze(1) for i, l in enumerate(self.lin)) + self.num_w(x["num"]).squeeze(1)
        return first + inter


# %% Step 3 — FFM｜列ごとに「相手の列の数」だけ埋め込みを持ち、組 (i, j) では v_{i,f(j)} と v_{j,f(i)} の内積を使う。
class FFM(nn.Module):
    def __init__(self, sizes, n_num, k=4):
        super().__init__()
        F = len(sizes)
        self.F = F
        self.emb = nn.ModuleList([nn.Embedding(s, F * k) for s in sizes])
        self.lin = nn.ModuleList([nn.Embedding(s, 1) for s in sizes])
        self.num_w = nn.Linear(n_num, 1)
        self.k = k
        self.iu = torch.triu_indices(F, F, 1)
        init_(self)

    def forward(self, x):
        V = torch.stack([e(x["cat"][:, i]).view(-1, self.F, self.k) for i, e in enumerate(self.emb)], 1)   # (行, 列 i, 相手 f, k)
        i, j = self.iu
        inter = (V[:, i, j] * V[:, j, i]).sum(-1).sum(1)
        first = sum(l(x["cat"][:, c]).squeeze(1) for c, l in enumerate(self.lin)) + self.num_w(x["num"]).squeeze(1)
        return first + inter


# %% Step 4 — Wide & Deep と DeepFM｜Wide（1 次）＋ Deep（埋め込みをつないだ多層パーセプトロン）。DeepFM は Wide の代わりに FM の項を使う。
def mlp(d_in, hidden, dropout):
    layers = []
    for h in hidden:
        layers += [nn.Linear(d_in, h), nn.ReLU(), nn.Dropout(dropout)]
        d_in = h
    return nn.Sequential(*layers, nn.Linear(d_in, 1))


class WideDeep(FM):
    def __init__(self, sizes, n_num, k=8, hidden=(128, 64), dropout=0.1, fm_term=False):
        super().__init__(sizes, n_num, k)
        self.deep = mlp(len(sizes) * k + n_num, hidden, dropout)
        self.fm_term = fm_term
        init_(self)

    def forward(self, x):
        E = torch.stack([e(x["cat"][:, i]) for i, e in enumerate(self.emb)], 1)
        out = self.deep(torch.cat([E.flatten(1), x["num"]], 1)).squeeze(1)
        out = out + sum(l(x["cat"][:, i]).squeeze(1) for i, l in enumerate(self.lin)) + self.num_w(x["num"]).squeeze(1)
        if self.fm_term:                                                    # DeepFM：埋め込みを Deep と FM で共有する
            V = self.vectors(x)
            out = out + 0.5 * ((V.sum(1) ** 2) - (V ** 2).sum(1)).sum(1)
        return out


class DeepFM(WideDeep):
    def __init__(self, sizes, n_num, k=8, hidden=(128, 64), dropout=0.1):
        super().__init__(sizes, n_num, k, hidden, dropout, fm_term=True)


# %% Step 5 — DIN｜閲覧の系列の各商品（商品＋カテゴリの埋め込み）と、表示した商品の関係（差・積）から注意の重みを計算し、
#               系列を重み付きで足して「この商品に対する会員の関心」を作る。
class DIN(WideDeep):
    def __init__(self, sizes, n_num, k=8, hidden=(128, 64), dropout=0.1, att_hidden=36):
        super().__init__(sizes, n_num, k, hidden, dropout)
        self.item_f, self.cat_f = FIELDS.index("item_id"), FIELDS.index("category_id")
        self.att = nn.Sequential(nn.Linear(8 * k, att_hidden), nn.ReLU(), nn.Linear(att_hidden, 1))
        self.deep = mlp(len(sizes) * k + 2 * k + n_num, hidden, dropout)
        init_(self)

    def forward(self, x):
        E = torch.stack([e(x["cat"][:, i]) for i, e in enumerate(self.emb)], 1)
        tgt = torch.cat([E[:, self.item_f], E[:, self.cat_f]], -1)          # 表示した商品（商品とカテゴリ）
        H = torch.cat([self.emb[self.item_f](x["seq"]), self.emb[self.cat_f](x["seqc"])], -1)   # 系列 (行, 20, 2k)
        T = tgt.unsqueeze(1).expand_as(H)
        a = self.att(torch.cat([H, T, H - T, H * T], -1)).squeeze(-1)       # 注意の点数（softmax にしない＝関心の強さを残す、DIN の原論文と同じ）
        a = a.masked_fill(x["seq"] == 0, 0.0)
        interest = (a.unsqueeze(-1) * H).sum(1)
        out = self.deep(torch.cat([E.flatten(1), interest, x["num"]], 1)).squeeze(1)
        return out + sum(l(x["cat"][:, i]).squeeze(1) for i, l in enumerate(self.lin)) + self.num_w(x["num"]).squeeze(1)


# %% Step 6 — SIM（縮小版）｜長い系列（100 件）から、表示した商品と同じカテゴリの閲覧だけを「検索」で選び（hard search）、
#               選んだ閲覧に注意を掛けて長期の関心を作る。直近 20 件の DIN の関心と合わせて使う。
class SIM(DIN):
    def __init__(self, sizes, n_num, k=8, hidden=(128, 64), dropout=0.1, att_hidden=36):
        super().__init__(sizes, n_num, k, hidden, dropout, att_hidden)
        self.att_long = nn.Sequential(nn.Linear(8 * k, att_hidden), nn.ReLU(), nn.Linear(att_hidden, 1))
        self.deep = mlp(len(sizes) * k + 4 * k + n_num, hidden, dropout)
        init_(self)

    def interest(self, att, H, tgt, mask):
        T = tgt.unsqueeze(1).expand_as(H)
        a = att(torch.cat([H, T, H - T, H * T], -1)).squeeze(-1).masked_fill(~mask, 0.0)
        return (a.unsqueeze(-1) * H).sum(1)

    def forward(self, x):
        E = torch.stack([e(x["cat"][:, i]) for i, e in enumerate(self.emb)], 1)
        tgt = torch.cat([E[:, self.item_f], E[:, self.cat_f]], -1)
        H = torch.cat([self.emb[self.item_f](x["seq"]), self.emb[self.cat_f](x["seqc"])], -1)
        L = torch.cat([self.emb[self.item_f](x["lseq"]), self.emb[self.cat_f](x["lseqc"])], -1)
        same_cat = (x["lseqc"] == x["cat"][:, self.cat_f].unsqueeze(1)) & (x["lseq"] > 0)   # hard search：同じカテゴリだけ
        short = self.interest(self.att, H, tgt, x["seq"] > 0)
        long = self.interest(self.att_long, L, tgt, same_cat)
        out = self.deep(torch.cat([E.flatten(1), short, long, x["num"]], 1)).squeeze(1)
        return out + sum(l(x["cat"][:, i]).squeeze(1) for i, l in enumerate(self.lin)) + self.num_w(x["num"]).squeeze(1)


# %% Step 7 — 学習ループ｜Adam・バッチ 4,096・対数損失。エポックごとに検証の LogLoss を返し（Optuna の枝刈りに使う）、予測を返す。
def reinit(model, seed):
    """乱数の種を決めてから、すべての重みを初期値に戻す（モデルを作った時点の乱数に結果が左右されないように）。"""
    torch.manual_seed(seed)
    for m in model.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    for name, p in model.named_parameters():
        if name.endswith("num_v"):
            nn.init.normal_(p, std=0.01)
    init_(model)


def train(model, xtr, ytr, xva, epochs=2, lr=2e-3, weight_decay=0.0, batch=4096, seed=0, report=None):
    reinit(model, seed)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    y = torch.from_numpy(ytr.astype(np.float32))
    n = len(y)
    loss_fn = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        for s in range(0, n, batch):
            i = perm[s:s + batch]
            opt.zero_grad()
            loss = loss_fn(model({k: v[i] for k, v in xtr.items()}), y[i])
            loss.backward()
            opt.step()
        if report is not None:
            report(ep, predict(model, xva))
    return predict(model, xva)


@torch.no_grad()
def predict(model, x, batch=65536):
    model.eval()
    n = len(x["cat"])
    out = [torch.sigmoid(model({k: v[s:s + batch] for k, v in x.items()})) for s in range(0, n, batch)]
    return torch.cat(out).numpy().astype(np.float64)
