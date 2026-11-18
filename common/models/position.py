"""第8章 8.7：表示順位の偏りを学習から外す最小実装（PyTorch）。PAL 型：p(クリック) = p(見られる | 順位) × p(クリック | 見られた, 特徴量)。"""
# %% Step 1 — PAL 型のモデル｜関心の塔（第8章の DeepFM。順位は入れない）と、順位だけの小さな塔（順位ごとに 1 つの値）の確率を掛ける。
#               学習は積に対数損失を掛け、推薦の採点には関心の塔だけを使う（順位に依らない確率）。
#               2 つの塔の確率は、掛けた値しか決まらない（一方を 2 倍、他方を 1/2 にしても同じ）。確率の大きさが要るときは、
#               関心の塔の確率に 1 位の見られやすさを掛けて「1 位に置いたときの確率」として使う。
from __future__ import annotations

import torch
from torch import nn

from common.models.torch_ctr import DeepFM


class PAL(nn.Module):
    def __init__(self, sizes, n_num, n_pos=7, k=8):
        super().__init__()
        self.rel = DeepFM(sizes, n_num, k=k)
        self.pos = nn.Embedding(n_pos, 1)                      # 順位 1〜6 の「見られやすさ」（0 は使わない）
        self.relevance_only = False

    def forward(self, x):
        lr = self.rel(x)
        if self.relevance_only:
            return lr                                          # 採点：順位に依らない確率の logit
        p = torch.sigmoid(lr) * torch.sigmoid(self.pos(x["pos"]).squeeze(1))
        p = p.clamp(1e-7, 1 - 1e-7)
        return torch.log(p) - torch.log1p(-p)                  # 積の確率を logit に直す（学習ループの対数損失に渡すため）

    def seen(self):
        """順位ごとの見られやすさ σ(順位の値)。"""
        with torch.no_grad():
            return torch.sigmoid(self.pos.weight[1:, 0]).numpy()
