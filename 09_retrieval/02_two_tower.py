"""第9章：two-tower（最小実装、PyTorch）。会員の塔と商品の塔で同じ空間のベクトルを作り、内積の大きい商品を候補にする。"""
# %% Step 1 — 学習の組｜1 週ずつ区切り、その週の行動（会員, 商品）を正例に、会員の履歴はその週の前日までの直近 20 件にする（時点を守る）。
import common  # noqa: F401  先に読み込む（LightGBM を PyTorch より先に。付録 A）
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from common import SEED, connect, show
from common.retrieval import eval_users, interactions, recall, true_recall, true_top

torch.manual_seed(SEED)
torch.set_num_threads(1)                                  # 並列の足し合わせの順で結果が揺れないよう、1 スレッドで学習する
con = connect()
items = con.sql("SELECT item_id, category_id, brand_id, gender FROM item ORDER BY item_id").df()
users_t = con.sql('SELECT user_id, gender, age_band, body_size FROM "user" ORDER BY user_id').df()
code = lambda s: pd.factorize(s, sort=True)[0] + 1
IT = torch.tensor(np.column_stack([items.item_id, items.category_id, items.brand_id, code(items.gender)]))
US = torch.tensor(np.column_stack([code(users_t.gender), code(users_t.age_band), code(users_t.body_size)]))


def history(until: int, n: int = 20) -> np.ndarray:
    h = interactions(28, until).sort_values(["user_id", "last_day", "item_id"], kind="stable").groupby("user_id").tail(n)
    H = np.zeros((len(users_t) + 1, n), dtype=np.int64)
    for u, g in h.groupby("user_id"):
        H[u, -len(g):] = g.item_id.to_numpy()
    return H


weeks = list(range(56, 133, 7))                           # day 56〜132 を 11 週に区切る
H_w = torch.stack([torch.tensor(history(w - 1)) for w in weeks])   # 週ごとの履歴 (週, 会員, 20)
pairs = pd.concat([interactions(w, w + 6).assign(wk=k) for k, w in enumerate(weeks)], ignore_index=True)

# %% Step 2 — 2 つの塔｜商品の塔：商品・カテゴリ・ブランド・性別の埋め込み → 全結合。会員の塔：属性と履歴の商品の平均 → 全結合。
class Tower(nn.Module):
    def __init__(self, k=32, d=64):
        super().__init__()
        self.item = nn.Embedding(8001, k, padding_idx=0)
        self.cat, self.brand, self.ig = nn.Embedding(28, k), nn.Embedding(121, k), nn.Embedding(4, k)
        self.ug, self.age, self.size = nn.Embedding(3, k), nn.Embedding(7, k), nn.Embedding(6, k)
        self.f_item = nn.Sequential(nn.Linear(4 * k, 128), nn.ReLU(), nn.Linear(128, d))
        self.f_user = nn.Sequential(nn.Linear(4 * k, 128), nn.ReLU(), nn.Linear(128, d))

    def item_vec(self, i):
        x = IT[i - 1]
        v = torch.cat([self.item(x[:, 0]), self.cat(x[:, 1]), self.brand(x[:, 2]), self.ig(x[:, 3])], 1)
        return nn.functional.normalize(self.f_item(v), dim=1)

    def user_vec(self, u, H, wk=None):
        h = H[u] if wk is None else H[wk, u]
        m = (h > 0).float().unsqueeze(-1)
        hist = (self.item(h) * m).sum(1) / m.sum(1).clamp(min=1)
        x = US[u - 1]
        v = torch.cat([self.ug(x[:, 0]), self.age(x[:, 1]), self.size(x[:, 2]), hist], 1)
        return nn.functional.normalize(self.f_user(v), dim=1)


# %% Step 3 — 学習｜同じバッチの他の商品を負例にする softmax（in-batch negatives）。温度 τ = 0.05。人気の商品ほど負例に
#               選ばれやすい偏りを、log（商品の出現の割合）を引いて補正する（logQ 補正）。
model, tau = Tower(), 0.05
freq = np.bincount(pairs.item_id, minlength=8001)[1:] + 1
logq = torch.tensor(np.log(freq / freq.sum()), dtype=torch.float32)
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
u_all, i_all = torch.tensor(pairs.user_id.to_numpy()), torch.tensor(pairs.item_id.to_numpy())
w_all = torch.tensor(pairs.wk.to_numpy())
t = time.time()
for ep in range(5):
    perm = torch.randperm(len(u_all), generator=torch.Generator().manual_seed(SEED + ep))
    for s in range(0, len(perm), 2048):
        b = perm[s:s + 2048]
        logits = model.user_vec(u_all[b], H_w, w_all[b]) @ model.item_vec(i_all[b]).T / tau - logq[i_all[b] - 1]
        loss = nn.functional.cross_entropy(logits, torch.arange(len(b)))
        opt.zero_grad(); loss.backward(); opt.step()
print(f"[time] 学習 {time.time() - t:.1f} 秒（{len(pairs):,} 組 × 5 エポック）")

# %% Step 4 — 候補｜評価の会員の履歴を day 132 までで作り直し、全商品のベクトルとの内積の上位 100 件（総当たり）を候補にする。
users, truth = eval_users()
tt = true_top()
need = np.union1d(users, tt.user_id.unique())
H_ev = torch.tensor(history(132))
with torch.no_grad():
    IV = model.item_vec(torch.arange(1, 8001))
    UV = model.user_vec(torch.tensor(need), H_ev)
t = time.time()
top = torch.topk(UV @ IV.T, 100, dim=1).indices.numpy() + 1
print(f"[time] 総当たりの検索 {time.time() - t:.2f} 秒（{len(need):,} 人 × 8,000 商品）")
cands = {u: list(r) for u, r in zip(need, top)}
show(pd.Series({**recall({u: cands[u] for u in users}, truth), **true_recall(cands, tt)}, name="two-tower（最小実装）"),
     "two-tower の候補（評価 day 133〜146、合成データ）")
np.save("outputs/09_retrieval/item_vec.npy", IV.numpy())
np.save("outputs/09_retrieval/user_vec.npy", UV.numpy())
