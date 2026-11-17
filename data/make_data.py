"""本書の通し作例：架空のファッション EC の合成データを作る。

真のクリック確率・購入確率・返品確率・クーポンの介入効果を「答え合わせ用」の truth_* テーブルに分けて保存する。
学習・評価のコードは truth_* を読まない（答え合わせのスクリプトだけが読む）。式と係数は付録 B に載せる。

使い方:
  python data/make_data.py                       # 既定（会員 5 万人・商品 8,000 点・26 週）→ data/generated/default/
  python data/make_data.py --scale 0.1 --out data/generated/small
  python data/make_data.py --seq-strength 0.3 --out data/generated/weak_seq   # 系列効果を弱めた版（第11章）
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
START = np.datetime64("2025-04-07")          # 月曜日。26 週＝2025-04-07〜2025-10-05
N_DAYS = 182
SALE = (81, 97)                              # 夏のセール：2025-06-27〜2025-07-13（day 81〜97）
CAMPAIGNS = [                                # クーポン配信（無作為割付、割付確率 0.5）
    dict(campaign_id=1, name="春夏アイテム 10%オフクーポン", day=42, discount_rate=0.10),
    dict(campaign_id=2, name="秋冬先行 10%オフクーポン", day=126, discount_rate=0.10),
]
SIZE_APPAREL = ["XS", "S", "M", "L", "XL"]
SIZE_SHOES = ["23.0", "24.0", "25.0", "26.0", "27.0", "28.0"]

# 大分類, 中分類, サイズの型, 季節, 返品のしやすさ, 基準価格（円）, 対象の性別（U=どちらも）
CATEGORIES = [
    ("トップス", "Tシャツ", "apparel", "SS", 0.2, 3500, "U"),
    ("トップス", "シャツ・ブラウス", "apparel", "ALL", 0.4, 5500, "U"),
    ("トップス", "ニット", "apparel", "AW", 0.4, 7500, "U"),
    ("トップス", "パーカー・スウェット", "apparel", "ALL", 0.2, 6500, "U"),
    ("ボトムス", "パンツ", "apparel", "ALL", 0.7, 7000, "U"),
    ("ボトムス", "デニム", "apparel", "ALL", 0.9, 9000, "U"),
    ("ボトムス", "スカート", "apparel", "ALL", 0.6, 6500, "F"),
    ("ボトムス", "ショートパンツ", "apparel", "SS", 0.5, 4500, "U"),
    ("アウター", "ジャケット", "apparel", "ALL", 0.6, 15000, "U"),
    ("アウター", "コート", "apparel", "AW", 0.7, 22000, "U"),
    ("アウター", "ブルゾン", "apparel", "AW", 0.5, 13000, "U"),
    ("アウター", "ダウン", "apparel", "AW", 0.6, 25000, "U"),
    ("ワンピース", "ワンピース", "apparel", "ALL", 0.8, 9000, "F"),
    ("ワンピース", "セットアップ", "apparel", "ALL", 0.8, 14000, "U"),
    ("シューズ", "スニーカー", "shoes", "ALL", 0.9, 9000, "U"),
    ("シューズ", "ブーツ", "shoes", "AW", 1.0, 16000, "U"),
    ("シューズ", "サンダル", "shoes", "SS", 0.7, 6000, "U"),
    ("シューズ", "パンプス", "shoes", "ALL", 1.1, 8000, "F"),
    ("バッグ", "トートバッグ", "free", "ALL", 0.1, 6000, "U"),
    ("バッグ", "ショルダーバッグ", "free", "ALL", 0.1, 8000, "U"),
    ("バッグ", "リュック", "free", "ALL", 0.1, 9000, "U"),
    ("雑貨", "帽子", "apparel", "ALL", 0.3, 3500, "U"),
    ("雑貨", "財布", "free", "ALL", 0.0, 9000, "U"),
    ("雑貨", "アクセサリー", "free", "ALL", 0.1, 4000, "U"),
    ("雑貨", "ベルト", "apparel", "ALL", 0.3, 4000, "U"),
    ("ルームウェア", "ルームウェア", "apparel", "ALL", 0.2, 4500, "U"),
    ("ルームウェア", "靴下・レッグウェア", "free", "ALL", 0.0, 1200, "U"),
]
# カートの「併せ買い」で一緒に出しやすい大分類
COMPLEMENT = {
    "トップス": ["ボトムス", "アウター", "雑貨"], "ボトムス": ["トップス", "シューズ", "雑貨"],
    "アウター": ["トップス", "ボトムス", "雑貨"], "ワンピース": ["シューズ", "バッグ", "雑貨"],
    "シューズ": ["ルームウェア", "ボトムス", "雑貨"], "バッグ": ["雑貨", "シューズ", "トップス"],
    "雑貨": ["バッグ", "トップス", "ルームウェア"], "ルームウェア": ["ルームウェア", "雑貨", "トップス"],
}


@dataclass
class Params:
    seed: int = 20261116
    scale: float = 1.0
    n_users: int = 50_000
    n_items: int = 8_000
    n_brands: int = 120
    dim: int = 8
    seq_strength: float = 1.0      # 直近の閲覧系列の効果の強さ（0 で効果なし）
    eps_top: float = 0.15          # ロギング方策の探索率（トップ）
    eps_other: float = 0.20        # 同（関連商品・カート・メール）
    top_slots: int = 6
    related_slots: int = 4
    cart_slots: int = 4
    mail_slots: int = 4
    p_related: float = 0.3         # 閲覧のあと関連商品の枠を表示する確率
    activity_mean: float = 0.7     # 1 週間の平均セッション数
    bounce_ratio: float = 3.5      # 直帰セッション（外部から商品ページに来て 1 件見て帰る）の、通常のセッションに対する比
    state_sample: float = 0.05     # 真の確率を後から再計算できるよう状態を記録するセッションの割合（第4・10章）


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Gen:
    def __init__(self, p: Params):
        self.p = p
        self.cart_today = []
        self.r_state = np.random.default_rng(p.seed + 1)
        self.r_bounce = np.random.default_rng(p.seed + 2)  # 直帰セッションも別の乱数（推薦の記録と真の確率を変えない）
        self.state_rows = []
        self.r = np.random.default_rng(p.seed)
        self.nu = max(200, int(p.n_users * p.scale))
        self.ni = max(100, int(p.n_items * p.scale))
        self.nb = max(10, int(p.n_brands * min(1.0, max(p.scale, 0.25))))
        self.rows = {k: [] for k in ["session", "impression", "event", "order", "order_line",
                                     "truth_impression", "truth_order_line", "inventory", "item_availability"]}
        self.next_id = {k: 1 for k in ["session", "impression", "event", "order", "order_line"]}

    # ---------- マスタ ----------
    def masters(self):
        r, p = self.r, self.p
        cat = pd.DataFrame(CATEGORIES, columns=["major", "minor", "size_type", "season", "return_base",
                                                 "base_price", "gender_target"])
        cat.insert(0, "category_id", np.arange(1, len(cat) + 1))
        self.cat = cat
        nb = self.nb
        tier = r.choice([1, 2, 3], size=nb, p=[0.35, 0.45, 0.20])
        bpop = 1.0 / np.arange(1, nb + 1) ** 0.8
        bpop = r.permutation(bpop)
        bgender = r.choice(["M", "F", "U"], size=nb, p=[0.25, 0.40, 0.35])
        self.brand = pd.DataFrame({"brand_id": np.arange(1, nb + 1), "brand_name": [f"ブランド{i:03d}" for i in range(1, nb + 1)],
                                   "price_tier": tier, "gender_focus": bgender})
        self.brand_pop = bpop / bpop.sum()

        ni = self.ni
        cw = np.array([3, 3, 2, 2, 3, 2, 2, 1.5, 1.5, 1.2, 1.2, 1, 2, 1, 2, 1.2, 1.2, 1.2, 1.5, 1.5, 1, 1, 1, 1.5, 0.8, 1, 1.2])
        icat = r.choice(len(cat), size=ni, p=cw / cw.sum())
        ibrand = r.choice(nb, size=ni, p=self.brand_pop)
        cat_g = cat.gender_target.values[icat]
        br_g = bgender[ibrand]
        g = np.where(cat_g == "F", "F", np.where(br_g == "U", r.choice(["M", "F", "U"], size=ni, p=[0.3, 0.4, 0.3]), br_g))
        tmult = np.array([0.7, 1.0, 1.6])[tier[ibrand] - 1]
        price = np.exp(r.normal(np.log(cat.base_price.values[icat] * tmult), 0.3))
        price = np.maximum(500, np.round(price / 100) * 100).astype(np.int64)
        season = cat.season.values[icat]
        rel = np.where(r.random(ni) < 0.45, -r.integers(1, 365, ni), 0)
        newd = np.where(season == "AW", r.integers(95, N_DAYS, ni),
                        np.where(season == "SS", r.integers(0, 80, ni), r.integers(0, N_DAYS, ni)))
        rel = np.where(rel < 0, rel, newd)
        # 潜在属性：カテゴリの重心＋ブランドのずれ＋個別
        ccent = r.normal(0, 1, (len(cat), p.dim))
        bcent = r.normal(0, 0.6, (nb, p.dim))
        ivec = (ccent[icat] * 0.6 + bcent[ibrand] * 0.5 + r.normal(0, 0.6, (ni, p.dim))) / np.sqrt(p.dim)
        self.item = pd.DataFrame({"item_id": np.arange(1, ni + 1), "category_id": icat + 1, "brand_id": ibrand + 1,
                                  "gender": g, "list_price": price, "season": season,
                                  "release_date": START + rel.astype("timedelta64[D]")})
        self.ivec, self.icat, self.ibrand, self.iprice, self.irel = ivec, icat, ibrand, price, rel
        self.iseason = season
        self.igender = g
        cost_ratio = np.array([0.55, 0.50, 0.45])[tier[ibrand] - 1]
        self.icost = np.round(price * cost_ratio).astype(np.int64)
        self.item["unit_cost"] = self.icost

        # SKU（サイズ・色）
        st = cat.size_type.values[icat]
        sku_rows = []
        sid = 1
        for i in range(ni):
            if st[i] == "apparel":
                sizes = list(range(5)) if r.random() < 0.6 else list(range(1, 4 + (r.random() < 0.5)))
            elif st[i] == "shoes":
                sizes = list(range(6)) if g[i] == "U" else (list(range(0, 4)) if g[i] == "F" else list(range(2, 6)))
            else:
                sizes = [-1]
            ncol = r.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
            for c in range(ncol):
                for s in sizes:
                    sku_rows.append((sid, i + 1, s, c))
                    sid += 1
        sku = pd.DataFrame(sku_rows, columns=["sku_id", "item_id", "size_idx", "color_idx"])
        lab = []
        for t, s in zip(st[sku.item_id.values - 1], sku.size_idx.values):
            lab.append("F" if s < 0 else (SIZE_SHOES[s] if t == "shoes" else SIZE_APPAREL[s]))
        sku["size"] = lab
        sku["color"] = np.array(["ブラック", "ホワイト", "ネイビー", "ベージュ", "グレー", "カーキ"])[
            (sku.color_idx.values + sku.item_id.values) % 6]
        self.sku = sku
        self.sku_item = sku.item_id.values - 1
        self.sku_size = sku.size_idx.values
        # 初期在庫：中心のサイズほど多い。発売前の商品は発売日に入荷
        center = np.where(st[self.sku_item] == "shoes", 2.5, 2.0)
        w = np.exp(-0.5 * ((self.sku_size - center) / 1.3) ** 2)
        w = np.where(self.sku_size < 0, 1.5, w)
        self.stock0 = r.poisson(8 * w + 1)
        self.stock = np.where(rel[self.sku_item] <= 0, self.stock0, 0)
        self.out_days = np.zeros(len(sku), dtype=np.int32)

        # ユーザー
        nu = self.nu
        gender = r.choice(["M", "F"], size=nu, p=[0.42, 0.58])
        reg = np.where(r.random(nu) < 0.72, -r.integers(1, 1000, nu), r.integers(0, N_DAYS, nu))
        age = r.choice(["10代", "20代", "30代", "40代", "50代", "60代以上"], size=nu, p=[0.06, 0.30, 0.28, 0.20, 0.11, 0.05])
        region = r.choice(["北海道・東北", "関東", "中部", "近畿", "中国・四国", "九州・沖縄"], size=nu, p=[0.1, 0.38, 0.16, 0.18, 0.08, 0.10])
        device = r.choice(["app", "sp_web", "pc_web"], size=nu, p=[0.5, 0.35, 0.15])
        loyalty = r.normal(0, 1, nu)
        act = np.exp(r.normal(np.log(p.activity_mean) - 0.5 * 0.8 ** 2, 0.8, nu) + 0.25 * loyalty)
        rank = np.digitize(loyalty + r.normal(0, 0.5, nu), [-0.3, 0.6, 1.4])   # 0..3
        size_app = np.clip(np.round(np.where(gender == "F", r.normal(1.4, 0.8, nu), r.normal(2.6, 0.8, nu))), 0, 4).astype(int)
        size_shoe = np.clip(np.round(np.where(gender == "F", r.normal(1.4, 0.9, nu), r.normal(3.8, 0.9, nu))), 0, 5).astype(int)
        price_level = r.normal(np.log(6500), 0.35, nu) + 0.15 * loyalty
        sens = np.exp(r.normal(0, 0.35, nu))
        uvec = r.normal(0, 1, (nu, p.dim)) / np.sqrt(p.dim)
        # 好きなブランド（3 つ）
        gmatch = (bgender[None, :] == gender[:, None]) | (bgender[None, :] == "U")
        fav = np.zeros((nu, nb), dtype=bool)
        for u in range(nu):
            w = self.brand_pop * np.where(gmatch[u], 1.0, 0.15)
            fav[u, r.choice(nb, size=3, replace=False, p=w / w.sum())] = True
        churn_risk = r.normal(0, 1, nu) - 0.4 * loyalty
        hazard = sigmoid(-3.6 + 0.6 * churn_risk) / 7.0           # 1 日あたり
        optin = r.random(nu) < 0.45
        self.user = pd.DataFrame({"user_id": np.arange(1, nu + 1), "gender": gender, "age_band": age, "region": region,
                                  "registered_at": START + reg.astype("timedelta64[D]"), "member_rank": np.array(
                                      ["ブロンズ", "シルバー", "ゴールド", "プラチナ"])[rank],
                                  "body_size": np.array(SIZE_APPAREL)[size_app], "shoe_size": np.array(SIZE_SHOES)[size_shoe],
                                  "device": device, "mail_opt_in": optin})
        self.u = dict(gender=gender, reg=reg, act=act, rank=rank, size_app=size_app, size_shoe=size_shoe, pl=price_level,
                      sens=sens, vec=uvec, fav=fav, hazard=hazard, churn_risk=churn_risk, loyalty=loyalty, optin=optin)
        self.churned = np.zeros(nu, dtype=bool)
        self.churn_day = np.full(nu, -1)
        self.cat_recent = np.zeros((nu, len(cat)), dtype=np.float32)
        self.brand_recent = np.zeros((nu, nb), dtype=np.float32)
        self.pop = np.zeros(ni)

    # ---------- 真の確率 ----------
    def season_fit(self, items, d):
        s = self.iseason[items]
        t = d / N_DAYS
        ss = np.clip(1.2 - 1.6 * max(0.0, t - 0.55) / 0.45, -0.6, 1.0)
        aw = np.clip(-0.8 + 2.4 * max(0.0, t - 0.5) / 0.5, -0.8, 1.0)
        return np.where(s == "SS", ss, np.where(s == "AW", aw, 0.2))

    def refresh_size_stock(self):
        """商品×サイズの在庫の有無（列 0〜5 がサイズ、列 6 がサイズの無い商品）。1 日の始めに更新する。"""
        a = np.zeros((self.ni, 7), dtype=bool)
        col = np.where(self.sku_size < 0, 6, self.sku_size)
        ok = self.stock > 0
        a[self.sku_item[ok], col[ok]] = True
        self.size_stock = a

    def size_available(self, users, items):
        """ユーザーのサイズの SKU に在庫があるか（サイズの無い商品は在庫の有無）。"""
        st = self.cat.size_type.values[self.icat[items]]
        want = np.where(st == "shoes", self.u["size_shoe"][users], np.where(st == "apparel", self.u["size_app"][users], 6))
        return self.size_stock[items, want]

    def probs(self, users, items, d, surface, position, hour):
        u, p = self.u, self.p
        dot = np.einsum("ij,ij->i", u["vec"][users], self.ivec[items]) * 4.0
        fav = u["fav"][users, self.ibrand[items]].astype(float)
        seqc = np.log1p(self.cat_recent[users, self.icat[items]])
        seqb = np.log1p(self.brand_recent[users, self.ibrand[items]])
        price_term = u["sens"][users] * (np.log(self.iprice[items]) - u["pl"][users])
        new = ((d - self.irel[items]) >= 0) & ((d - self.irel[items]) < 14)
        sale = (SALE[0] <= d <= SALE[1]) & (self.iseason[items] == "SS")
        gmis = (self.igender[items] != "U") & (self.igender[items] != u["gender"][users])
        off = {"top": 0.0, "related": 0.0, "cart": -0.5, "mail": -1.2}[surface]
        night = ((hour >= 21) | (hour <= 1)).astype(float)
        lc = (-5.0 + off + 0.9 * dot + 0.8 * fav + p.seq_strength * (0.55 * seqc + 0.35 * seqb)
              + 0.6 * self.season_fit(items, d) + 0.5 * new + 0.4 * sale - 0.15 * (position - 1)
              - 0.7 * price_term - 1.5 * gmis + 0.15 * night)
        avail = self.size_available(users, items)
        lb = (-3.7 + 0.6 * dot + 0.5 * fav + p.seq_strength * 0.25 * seqc - 2.8 * (~avail)
              - 0.9 * price_term + 0.8 * sale + 0.2 * u["rank"][users])
        return sigmoid(lc), sigmoid(lb), avail

    # ---------- ロギング方策 ----------
    def policy(self, pool, n, eps, pop_w, k_top=100):
        """pool から n 枠を独立に選ぶ。各枠の表示確率（傾向スコア）＝ eps/|pool| ＋ (1-eps)×人気上位での重み。"""
        r = self.r
        m = len(pool)
        order = np.argsort(-pop_w[pool], kind="stable")[:min(k_top, m)]
        top = pool[order]
        w = pop_w[top] + 1.0
        w = w / w.sum()
        explore = r.random(n) < eps
        pick = np.where(explore, pool[r.integers(0, m, n)], top[r.choice(len(top), size=n, p=w)])
        o = np.argsort(top)
        k = np.clip(np.searchsorted(top[o], pick), 0, len(top) - 1)
        hit = top[o][k] == pick
        prop = eps / m + (1 - eps) * np.where(hit, w[o][k], 0.0)
        return pick, prop

    # ---------- 1 日の流れ ----------
    def run(self):
        r, p, u = self.r, self.p, self.u
        self.sku_index = [[] for _ in range(self.ni)]
        for k, it in enumerate(self.sku_item):
            self.sku_index[it].append(k)
        self.sku_index = [np.array(v) for v in self.sku_index]
        cat_major = self.cat.major.values
        decay = np.exp(-1 / 7)
        hour_p = np.array([2, 1, .5, .3, .3, .5, 1, 2, 3, 3, 3, 4, 6, 5, 4, 4, 4, 5, 6, 7, 8, 9, 9, 6], float)
        hour_p /= hour_p.sum()
        self.campaign_rows = []
        for d in range(N_DAYS):
            date = START + np.timedelta64(d, "D")
            wd = int((d) % 7)                                    # 0=月
            # 入荷・在庫切れの補充
            arriving = np.where(self.irel[self.sku_item] == d)[0]
            self.stock[arriving] = self.stock0[arriving]
            self.out_days = np.where(self.stock <= 0, self.out_days + 1, 0)
            restock = (self.out_days >= 7) & (r.random(len(self.stock)) < 0.25) & (self.irel[self.sku_item] <= d)
            self.stock[restock] = r.poisson(6, restock.sum()) + 1
            # 離反
            alive = (u["reg"] <= d) & ~self.churned
            newly = alive & (r.random(self.nu) < u["hazard"])
            self.churned |= newly
            self.churn_day[newly] = d
            alive &= ~newly
            # 表示できる商品（発売済みで在庫あり）
            avail_item = np.zeros(self.ni, dtype=bool)
            np.logical_or.at(avail_item, self.sku_item, self.stock > 0)
            avail_item &= self.irel <= d
            pools = {g: np.where(avail_item & ((self.igender == g) | (self.igender == "U")))[0] for g in ["M", "F"]}
            # 表示できる商品の日次の記録（推薦サーバーが参照する「その日に出せる商品」。方策の評価に使う）
            self.rows["item_availability"].append(pd.DataFrame({"item_id": np.where(avail_item)[0] + 1, "date": date}))
            cat_pools = {c: np.where(avail_item & (self.icat == c))[0] for c in range(len(self.cat))}
            maj_pools = {}
            for mj, comp in COMPLEMENT.items():
                inm = np.isin(cat_major[self.icat], comp) & avail_item
                for g in ["M", "F"]:
                    maj_pools[(mj, g)] = np.where(inm & ((self.igender == g) | (self.igender == "U")))[0]
            self.refresh_size_stock()
            # クーポン（無作為割付）
            for c in CAMPAIGNS:
                if d == c["day"]:
                    self.assign_campaign(c, alive, d)
            # セッション
            boost = (1.35 if SALE[0] <= d <= SALE[1] else 1.0) * (1.15 if wd >= 5 else 1.0)
            n_s = r.poisson(u["act"] / 7 * boost * alive)
            users = np.repeat(np.arange(self.nu), n_s)
            if len(users) == 0:
                continue
            hours = r.choice(24, size=len(users), p=hour_p)
            secs = hours * 3600 + r.integers(0, 3600, len(users))
            sids = self.new_ids("session", len(users))
            dev = u_dev = self.user.device.values[users]
            self.rows["session"].append(pd.DataFrame({"session_id": sids, "user_id": users + 1,
                                                      "started_at": date + secs.astype("timedelta64[s]"),
                                                      "device": u_dev}))
            # 直帰セッション：外部（検索エンジン・広告）から商品ページに来て 1 件だけ見て帰る。推薦枠は出ず、購入もしない。
            # 閲覧は記録に残るが、会員の嗜好の状態（直近の閲覧の記録）は変えない（滞在が短いため）
            rb = self.r_bounce
            n_b = rb.poisson(u["act"] / 7 * boost * alive * p.bounce_ratio)
            bu_ = np.repeat(np.arange(self.nu), n_b)
            if len(bu_):
                bsec = rb.choice(24, size=len(bu_), p=hour_p) * 3600 + rb.integers(0, 3600, len(bu_))
                bsid = self.new_ids("session", len(bu_))
                self.rows["session"].append(pd.DataFrame({"session_id": bsid, "user_id": bu_ + 1,
                                                          "started_at": date + bsec.astype("timedelta64[s]"),
                                                          "device": self.user.device.values[bu_]}))
                bi_ = np.empty(len(bu_), dtype=np.int64)
                for g in ["M", "F"]:
                    mg = u["gender"][bu_] == g
                    if mg.any() and len(pools[g]):
                        w = self.pop[pools[g]] + 1.0
                        bi_[mg] = pools[g][rb.choice(len(pools[g]), size=mg.sum(), p=w / w.sum())]
                self.add_events(bu_, bi_, bsid, bsec + 2, "view", "landing", None, date)
            # 状態の記録（トップの表示の直前。同じ日のトップの表示は、この状態で確率を計算する）
            sm = self.r_state.random(len(users)) < p.state_sample
            if sm.any():
                self.state_rows.append(pd.DataFrame({
                    "session_id": sids[sm], "user_id": users[sm] + 1, "day": d, "hour": ((secs[sm] + 5) // 3600) % 24,
                    "cat_recent": self.cat_recent[users[sm]].astype(np.float64).tolist(),
                    "brand_recent": self.brand_recent[users[sm]].astype(np.float64).tolist()}))
            # トップのおすすめ
            pending = []   # (user, item, surface, position, policy, prop, time, session)
            for g in ["M", "F"]:
                sel = np.where(u["gender"][users] == g)[0]
                if len(sel) == 0 or len(pools[g]) == 0:
                    continue
                n = len(sel) * p.top_slots
                items, prop = self.policy(pools[g], n, p.eps_top, self.pop)
                pending.append((np.repeat(users[sel], p.top_slots), items, "top", np.tile(np.arange(1, p.top_slots + 1), len(sel)),
                                "top_pop_eps", prop, np.repeat(secs[sel], p.top_slots) + 5, np.repeat(sids[sel], p.top_slots)))
            views = self.serve(pending, d, date, hour_of=lambda t: (t // 3600) % 24)
            # ブラウズ・検索による閲覧（推薦枠を経由しない）
            nv = r.poisson(2.5, len(users))
            vu = np.repeat(np.arange(len(users)), nv)
            if len(vu):
                bu = users[vu]
                cand = np.empty((len(vu), 40), dtype=np.int64)
                for g in ["M", "F"]:
                    m = u["gender"][bu] == g
                    if m.any() and len(pools[g]):
                        cand[m] = pools[g][r.integers(0, len(pools[g]), (m.sum(), 40))]
                sc = (np.einsum("ij,ikj->ik", u["vec"][bu], self.ivec[cand]) * 4.0 * 0.9
                      + 0.8 * u["fav"][bu[:, None], self.ibrand[cand]]
                      + p.seq_strength * 0.55 * np.log1p(self.cat_recent[bu[:, None], self.icat[cand]])
                      + 0.6 * self.season_fit(cand.ravel(), d).reshape(cand.shape))
                pr = np.exp(sc - sc.max(1, keepdims=True))
                pr /= pr.sum(1, keepdims=True)
                choice = (pr.cumsum(1) > r.random((len(vu), 1))).argmax(1)
                bi = cand[np.arange(len(vu)), choice]
                bt = secs[vu] + 60 + r.integers(0, 1200, len(vu))
                src = np.where(r.random(len(vu)) < 0.35, "search", "browse")
                srch = src == "search"
                self.add_events(bu[srch], None, sids[vu][srch], bt[srch] - 20, "search", "search", None, date,
                                category=self.icat[bi[srch]] + 1)
                self.add_events(bu, bi, sids[vu], bt, "view", src, None, date)
                self.bump_recent(bu, bi)
                # 閲覧からの購入
                _, pb, avail = self.probs(bu, bi, d, "related", np.ones(len(bu)), (bt // 3600) % 24)
                buy = r.random(len(bu)) < pb * 0.25
                self.purchase(bu[buy], bi[buy], sids[vu][buy], bt[buy] + 300, src[buy], None, d, date)
                views = views + [(bu, bi, bt, sids[vu])]
            # 関連商品（閲覧のあと）
            pending = []
            for (vu_, vi_, vt_, vs_) in views:
                show = r.random(len(vu_)) < p.p_related
                for k in np.where(show)[0]:
                    cpool = cat_pools[self.icat[vi_[k]]]
                    cpool = cpool[cpool != vi_[k]]
                    if len(cpool) < 5:
                        continue
                    it, pr_ = self.policy(cpool, p.related_slots, p.eps_other, self.pop, k_top=30)
                    pending.append((np.full(p.related_slots, vu_[k]), it, "related", np.arange(1, p.related_slots + 1),
                                    "related_pop_eps", pr_, np.full(p.related_slots, vt_[k] + 10), np.full(p.related_slots, vs_[k])))
            self.serve(pending, d, date, hour_of=lambda t: (t // 3600) % 24, make_related=False)
            # カートの併せ買い
            pending = []
            for (cu, ci, ct, cs) in self.cart_today:
                cpool = maj_pools[(cat_major[self.icat[ci]], u["gender"][cu])]
                if len(cpool) < 5:
                    continue
                it, pr_ = self.policy(cpool, p.cart_slots, p.eps_other, self.pop, k_top=60)
                pending.append((np.full(p.cart_slots, cu), it, "cart", np.arange(1, p.cart_slots + 1), "cart_pop_eps", pr_,
                                np.full(p.cart_slots, ct + 5), np.full(p.cart_slots, cs)))
            self.cart_today = []
            self.serve(pending, d, date, hour_of=lambda t: (t // 3600) % 24, make_related=False)
            self.cart_today = []
            # メール（月曜、配信許諾のある会員）
            if wd == 0:
                mu = np.where(alive & u["optin"])[0]
                pending = []
                for g in ["M", "F"]:
                    sel = mu[u["gender"][mu] == g]
                    if len(sel) == 0:
                        continue
                    it, pr_ = self.policy(pools[g], len(sel) * p.mail_slots, p.eps_other, self.pop)
                    pending.append((np.repeat(sel, p.mail_slots), it, "mail", np.tile(np.arange(1, p.mail_slots + 1), len(sel)),
                                    "mail_pop_eps", pr_, np.full(len(sel) * p.mail_slots, 8 * 3600), np.full(len(sel) * p.mail_slots, -1)))
                self.serve(pending, d, date, hour_of=lambda t: (t // 3600) % 24, make_related=False)
                self.cart_today = []
            # 日次の更新
            self.cat_recent *= decay
            self.brand_recent *= decay
            self.pop *= decay
            if d % 7 == 6 or d == N_DAYS - 1:
                self.rows["inventory"].append(pd.DataFrame({"sku_id": self.sku.sku_id.values, "snapshot_date": date,
                                                            "stock": self.stock.astype(np.int32)}))
        self.finish_campaigns()

    def new_ids(self, kind, n):
        s = self.next_id[kind]
        self.next_id[kind] += n
        return np.arange(s, s + n, dtype=np.int64)

    def bump_recent(self, users, items):
        np.add.at(self.cat_recent, (users, self.icat[items]), 1.0)
        np.add.at(self.brand_recent, (users, self.ibrand[items]), 1.0)

    def add_events(self, users, items, sids, t, etype, source, imp, date, category=None):
        n = len(users)
        if n == 0:
            return
        ids = self.new_ids("event", n)
        self.rows["event"].append(pd.DataFrame({
            "event_id": ids, "user_id": users + 1, "session_id": np.where(np.asarray(sids) < 0, pd.NA, sids),
            "event_type": etype, "item_id": (items + 1) if items is not None else pd.NA,
            "category_id": category if category is not None else (self.icat[items] + 1 if items is not None else pd.NA),
            "source": source, "impression_id": imp if imp is not None else pd.NA,
            "occurred_at": date + np.asarray(t).astype("timedelta64[s]")}))

    def serve(self, pending, d, date, hour_of, make_related=True):
        """表示を記録し、真の確率でクリック・購入を引く。閲覧（クリック）のリストを返す。"""
        r = self.r
        views = []
        # 同じ推薦面の表示はまとめて処理する（小さな表を大量に作らないため）
        merged = {}
        for (uu, ii, surf, pos, pol, prop, t, ss) in pending:
            merged.setdefault((surf, pol), []).append((uu, ii, pos, prop, t, ss))
        pending = [(np.concatenate([x[0] for x in v]), np.concatenate([x[1] for x in v]), k[0],
                    np.concatenate([x[2] for x in v]), k[1], np.concatenate([x[3] for x in v]),
                    np.concatenate([x[4] for x in v]), np.concatenate([x[5] for x in v])) for k, v in merged.items()]
        for (uu, ii, surf, pos, pol, prop, t, ss) in pending:
            if len(uu) == 0:
                continue
            ids = self.new_ids("impression", len(uu))
            pc, pb, avail = self.probs(uu, ii, d, surf, pos, hour_of(t))
            self.rows["impression"].append(pd.DataFrame({
                "impression_id": ids, "user_id": uu + 1, "session_id": np.where(ss < 0, pd.NA, ss), "item_id": ii + 1,
                "surface": surf, "position": pos.astype(np.int16), "policy": pol, "propensity": prop,
                "shown_at": date + t.astype("timedelta64[s]")}))
            self.rows["truth_impression"].append(pd.DataFrame({"impression_id": ids, "p_click": pc,
                                                               "p_purchase_given_click": pb, "p_purchase": pc * pb}))
            click = r.random(len(uu)) < pc
            ct = t + 3 + r.integers(0, 30, len(uu))
            src = surf
            self.add_events(uu[click], ii[click], ss[click], ct[click], "view", src, ids[click], date)
            self.bump_recent(uu[click], ii[click])
            self.pop[ii[click]] += 0.2
            buy = click & (r.random(len(uu)) < pb)
            self.purchase(uu[buy], ii[buy], ss[buy], ct[buy] + 120, np.full(buy.sum(), surf), ids[buy], d, date)
            if make_related:
                views.append((uu[click], ii[click], ct[click], ss[click]))
        return views

    def purchase(self, users, items, sids, t, source, imp, d, date):
        r, u = self.r, self.u
        if len(users) == 0:
            return
        imp = imp if imp is not None else np.full(len(users), -1)
        keep_u, keep_i, keep_s, keep_t, keep_src, keep_imp, keep_sku, keep_mis = [], [], [], [], [], [], [], []
        st_all = self.cat.size_type.values
        for k in range(len(users)):
            uu, it = users[k], items[k]
            idx = self.sku_index[it]
            idx = idx[self.stock[idx] > 0]
            if len(idx) == 0:
                continue
            st = st_all[self.icat[it]]
            want = u["size_shoe"][uu] if st == "shoes" else (u["size_app"][uu] if st == "apparel" else -1)
            if want >= 0:
                dist = np.abs(self.sku_size[idx] - want)
                cand = idx[dist == dist.min()]
                mis = int(dist.min())
            else:
                cand, mis = idx, 0
            s = cand[r.integers(0, len(cand))]
            self.stock[s] -= 1
            keep_u.append(uu); keep_i.append(it); keep_s.append(sids[k]); keep_t.append(t[k]); keep_src.append(source[k])
            keep_imp.append(imp[k]); keep_sku.append(s); keep_mis.append(mis)
        if not keep_u:
            return
        uu = np.array(keep_u); ii = np.array(keep_i); ss = np.array(keep_s); tt = np.array(keep_t)
        src = np.array(keep_src); imps = np.array(keep_imp); skus = np.array(keep_sku); mis = np.array(keep_mis)
        impv = np.where(imps < 0, pd.NA, imps)
        self.add_events(uu, ii, ss, tt - 60, "cart", src, impv, date)
        self.add_events(uu, ii, ss, tt, "purchase", src, impv, date)
        self.cart_today.extend(zip(uu, ii, tt, ss))
        self.pop[ii] += 1.0
        # 注文：同じセッション（メールはユーザーと日）の購入を 1 注文にまとめる
        key = np.where(ss < 0, -(uu + 1), ss)
        _, inv = np.unique(key, return_inverse=True)
        oids = self.new_ids("order", inv.max() + 1)
        first = pd.DataFrame({"g": inv, "t": tt}).groupby("g").t.min().values
        self.rows["order"].append(pd.DataFrame({"order_id": oids, "user_id": pd.Series(uu).groupby(inv).first().values + 1,
                                                "session_id": pd.Series(np.where(ss < 0, pd.NA, ss)).groupby(inv).first().values,
                                                "ordered_at": date + first.astype("timedelta64[s]")}))
        lids = self.new_ids("order_line", len(uu))
        sale = (SALE[0] <= d <= SALE[1]) & (self.iseason[ii] == "SS")
        disc = np.where(sale, 0.3, 0.0)
        unit = np.round(self.iprice[ii] * (1 - disc)).astype(np.int64)
        self.rows["order_line"].append(pd.DataFrame({
            "order_line_id": lids, "order_id": oids[inv], "sku_id": skus + 1, "item_id": ii + 1, "quantity": 1,
            "list_price": self.iprice[ii], "discount_rate": disc, "unit_price": unit, "unit_cost": self.icost[ii],
            "source_surface": src, "impression_id": impv}))
        rb = self.cat.return_base.values[self.icat[ii]]
        new_user = (d - u["reg"][uu]) < 30
        pr = sigmoid(-2.1 + 1.5 * mis + 0.9 * rb + 0.3 * (np.log(self.iprice[ii]) - np.log(6500)) + 0.4 * new_user
                     - 0.25 * u["rank"][uu] + 0.3 * sale)
        self.rows["truth_order_line"].append(pd.DataFrame({"order_line_id": lids, "p_return": pr, "size_mismatch": mis}))

    # ---------- クーポン ----------
    def assign_campaign(self, c, alive, d):
        r, u = self.r, self.u
        el = np.where(alive & (u["reg"] < d - 14))[0]
        treat = r.random(len(el)) < 0.5
        cz = (u["churn_risk"][el] - u["churn_risk"].mean()) / u["churn_risk"].std()
        sz = (np.log(u["sens"][el]) - np.log(u["sens"]).mean()) / np.log(u["sens"]).std()
        az = (np.log(u["act"][el]) - np.log(u["act"]).mean()) / np.log(u["act"]).std()
        loyal = ((u["rank"][el] >= 2) & (az > 0.5)).astype(float)
        l0 = -2.0 + 0.9 * az + 0.3 * u["rank"][el] - 0.6 * cz
        bump = 0.50 + 0.35 * sz + 0.30 * cz - 0.9 * loyal
        p0, p1 = sigmoid(l0), sigmoid(l0 + bump)
        y = r.random(len(el)) < np.where(treat, p1, p0)
        self.campaign_rows.append(pd.DataFrame({
            "campaign_id": c["campaign_id"], "user_id": el + 1, "group": np.where(treat, "treatment", "control"),
            "assignment_prob": 0.5, "sent_at": START + np.timedelta64(c["day"], "D") + np.timedelta64(9, "h"),
            "purchased_14d": y, "coupon_used": y & treat, "p0": p0, "p1": p1}))

    def finish_campaigns(self):
        ca = pd.concat(self.campaign_rows, ignore_index=True)
        ca.insert(0, "assignment_id", np.arange(1, len(ca) + 1))
        self.truth_campaign = ca[["assignment_id", "p0", "p1"]].assign(uplift=lambda x: x.p1 - x.p0)
        self.campaign_assignment = ca.drop(columns=["p0", "p1"])
        self.campaign = pd.DataFrame([{**{k: v for k, v in c.items() if k != "day"},
                                       "kind": "coupon", "assignment_prob": 0.5,
                                       "start_date": START + np.timedelta64(c["day"], "D"),
                                       "end_date": START + np.timedelta64(c["day"] + 13, "D")} for c in CAMPAIGNS])

    # ---------- 保存 ----------
    def tables(self):
        t = {k: pd.concat(v, ignore_index=True) for k, v in self.rows.items() if v}
        t["event"] = t["event"].sort_values(["occurred_at", "event_id"], kind="stable").reset_index(drop=True)
        rets = t["order_line"][["order_line_id", "order_id"]].merge(t["truth_order_line"], on="order_line_id")
        rets = rets.merge(t["order"][["order_id", "ordered_at"]], on="order_id")
        ret = self.r.random(len(rets)) < rets.p_return.values
        rr = rets[ret]
        reason = np.where(rr.size_mismatch.values > 0, "サイズが合わない",
                          self.r.choice(["イメージと違う", "サイズが合わない", "品質", "その他"], size=len(rr), p=[0.45, 0.3, 0.1, 0.15]))
        t["return"] = pd.DataFrame({"return_id": np.arange(1, len(rr) + 1), "order_line_id": rr.order_line_id.values,
                                    "returned_at": rr.ordered_at.values + (self.r.integers(3, 15, len(rr)) * 86400).astype("timedelta64[s]"),
                                    "reason": reason})
        t["user"] = self.user
        t["category"] = self.cat.drop(columns=["return_base", "base_price"])
        t["brand"] = self.brand
        t["item"] = self.item
        t["sku"] = self.sku.drop(columns=["size_idx", "color_idx"])
        t["campaign"] = self.campaign
        t["campaign_assignment"] = self.campaign_assignment
        t["truth_campaign"] = self.truth_campaign
        u = self.u
        t["truth_user"] = pd.DataFrame({
            "user_id": np.arange(1, self.nu + 1), "churn_day": self.churn_day, "vec": u["vec"].tolist(),
            "fav_brands": [(np.where(r)[0] + 1).tolist() for r in u["fav"]], "sens": u["sens"], "price_level": u["pl"],
            "rank_idx": u["rank"], "size_app": u["size_app"], "size_shoe": u["size_shoe"]})
        t["truth_item"] = pd.DataFrame({"item_id": np.arange(1, self.ni + 1), "vec": self.ivec.tolist(), "release_day": self.irel})
        t["truth_session_state"] = pd.concat(self.state_rows, ignore_index=True)
        return t


def table_hash(df: pd.DataFrame) -> str:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object and len(df) and isinstance(df[c].iloc[0], (list, np.ndarray)):
            df[c] = df[c].map(lambda v: json.dumps(np.asarray(v).round(10).tolist()))
    return format(int(pd.util.hash_pandas_object(df, index=False).sum()) & 0xFFFFFFFFFFFF, "012x")


def generate(params: Params) -> dict[str, pd.DataFrame]:
    g = Gen(params)
    g.masters()
    g.run()
    return g.tables()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=Params.seed)
    ap.add_argument("--seq-strength", type=float, default=1.0)
    ap.add_argument("--out", default=str(HERE / "generated" / "default"))
    a = ap.parse_args()
    p = Params(seed=a.seed, scale=a.scale, seq_strength=a.seq_strength)
    t0 = time.time()
    tabs = generate(p)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"params": asdict(p), "start": str(START), "days": N_DAYS, "tables": {}}
    for name, df in sorted(tabs.items()):
        df.to_parquet(out / f"{name}.parquet", index=False)
        meta["tables"][name] = {"rows": len(df), "hash": table_hash(df)}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"出力先: {out}")
    for name, m in meta["tables"].items():
        print(f"  {name:20s} {m['rows']:>11,} 行")
    print(f"[time] 生成時間 {time.time() - t0:.1f} 秒")


if __name__ == "__main__":
    main()
