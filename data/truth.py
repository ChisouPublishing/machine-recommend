"""答え合わせ用：記録した状態から、任意の商品の真のクリック確率（トップのおすすめ）を計算する。

make_data.py の probs() のクリック確率の式と同じ。truth_session_state に記録したセッション（約 5%）について、
表示されなかった商品の確率も計算できる。オフライン方策評価（第4章）とバンディットの模擬（第10章）の答え合わせに使う。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from make_data import N_DAYS, SALE


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Truth:
    def __init__(self, data_dir, seq_strength: float = 1.0):
        rd = lambda n: pd.read_parquet(f"{data_dir}/{n}.parquet")
        tu, ti, it, us = rd("truth_user"), rd("truth_item"), rd("item"), rd("user")
        self.uvec = np.stack(tu.vec.values)
        self.fav = np.zeros((len(tu), int(it.brand_id.max())), dtype=bool)
        for k, fb in enumerate(tu.fav_brands.values):
            self.fav[k, np.asarray(fb) - 1] = True
        self.sens, self.pl = tu.sens.values, tu.price_level.values
        self.ugender = us.gender.to_numpy(dtype=object)
        self.ivec = np.stack(ti.vec.values)
        self.irel = ti.release_day.values
        self.icat, self.ibrand = it.category_id.values - 1, it.brand_id.values - 1
        self.iprice = it.list_price.to_numpy()
        self.iseason, self.igender = it.season.to_numpy(dtype=object), it.gender.to_numpy(dtype=object)
        self.state = rd("truth_session_state").set_index("session_id")
        self.seq = seq_strength

    def season_fit(self, items, d):
        t = d / N_DAYS
        ss = np.clip(1.2 - 1.6 * np.maximum(0.0, t - 0.55) / 0.45, -0.6, 1.0)
        aw = np.clip(-0.8 + 2.4 * np.maximum(0.0, t - 0.5) / 0.5, -0.8, 1.0)
        s = self.iseason[items]
        return np.where(s == "SS", ss, np.where(s == "AW", aw, 0.2))

    def p_click(self, session_ids, items, positions, surface_offset: float = 0.0) -> np.ndarray:
        """session_ids：長さ n、items：(n, k) の item_id、positions：(n, k)。真のクリック確率 (n, k)。"""
        st = self.state.loc[np.asarray(session_ids)]
        u = st.user_id.values - 1
        d = st.day.values[:, None]
        hour = st.hour.values[:, None]
        it = np.asarray(items) - 1
        cat_r = np.stack(st.cat_recent.values)
        br_r = np.stack(st.brand_recent.values)
        dot = np.einsum("nj,nkj->nk", self.uvec[u], self.ivec[it]) * 4.0
        fav = self.fav[u[:, None], self.ibrand[it]].astype(float)
        seqc = np.log1p(np.take_along_axis(cat_r, self.icat[it], 1))
        seqb = np.log1p(np.take_along_axis(br_r, self.ibrand[it], 1))
        price = self.sens[u][:, None] * (np.log(self.iprice[it]) - self.pl[u][:, None])
        new = ((d - self.irel[it]) >= 0) & ((d - self.irel[it]) < 14)
        sale = (SALE[0] <= d) & (d <= SALE[1]) & (self.iseason[it] == "SS")
        gmis = (self.igender[it] != "U") & (self.igender[it] != self.ugender[u][:, None])
        night = ((hour >= 21) | (hour <= 1)).astype(float)
        lc = (-5.0 + surface_offset + 0.9 * dot + 0.8 * fav + self.seq * (0.55 * seqc + 0.35 * seqb)
              + 0.6 * self.season_fit(it, d) + 0.5 * new + 0.4 * sale - 0.15 * (np.asarray(positions) - 1)
              - 0.7 * price - 1.5 * gmis + 0.15 * night)
        return sigmoid(lc)
