"""第9章：近似最近傍検索（ANN）。総当たりの内積検索と、faiss（HNSW・IVF）・hnswlib を、速さと再現率で比べる。"""
# %% Step 1 — ベクトル｜02 で学習した商品のベクトル（8,000 件）を、雑音を足して 30 倍（24 万件）に増やし、大きな品ぞろえを模す。
import time

import faiss
import hnswlib
import numpy as np
import pandas as pd

from common import SEED, show

faiss.omp_set_num_threads(1)                              # 索引を作る順番を固定し、再現率を実行ごとに同じにする
r = np.random.default_rng(SEED)
IV = np.load("outputs/09_retrieval/item_vec.npy").astype(np.float32)
UV = np.load("outputs/09_retrieval/user_vec.npy").astype(np.float32)[:1000]
big = np.repeat(IV, 30, axis=0) + r.normal(0, 0.05, (len(IV) * 30, IV.shape[1])).astype(np.float32)
big /= np.linalg.norm(big, axis=1, keepdims=True)
K, rows = 100, []
print(f"商品のベクトル {big.shape[0]:,} 件 × {big.shape[1]} 次元、検索する会員 {len(UV):,} 人、K = {K}")

# %% Step 2 — 総当たり｜すべての商品との内積を計算して上位 K 件。これを正解として、ANN の再現率を測る。
t = time.time()
exact = faiss.IndexFlatIP(big.shape[1]); exact.add(big)
build = time.time() - t
t = time.time()
_, truth = exact.search(UV, K)
rows.append(("総当たり（faiss IndexFlatIP）", build, time.time() - t, 1.0))


def rec(I):
    return float(np.mean([len(set(a) & set(b)) / K for a, b in zip(I, truth)]))


# %% Step 3 — faiss の HNSW と IVF｜HNSW はグラフをたどる。IVF は 1,024 の区画に分け、近い区画（nprobe 個）だけを調べる。
for M_, ef in ((16, 64), (32, 128)):
    t = time.time()
    ix = faiss.IndexHNSWFlat(big.shape[1], M_, faiss.METRIC_INNER_PRODUCT); ix.hnsw.efConstruction = 80; ix.add(big)
    b = time.time() - t
    ix.hnsw.efSearch = ef
    t = time.time(); _, I = ix.search(UV, K)
    rows.append((f"faiss HNSW（M={M_}, efSearch={ef}）", b, time.time() - t, rec(I)))
t = time.time()
q = faiss.IndexFlatIP(big.shape[1])
ivf = faiss.IndexIVFFlat(q, big.shape[1], 1024, faiss.METRIC_INNER_PRODUCT); ivf.train(big[::10]); ivf.add(big)
b = time.time() - t
for nprobe in (8, 32):
    ivf.nprobe = nprobe
    t = time.time(); _, I = ivf.search(UV, K)
    rows.append((f"faiss IVF（nlist=1024, nprobe={nprobe}）", b, time.time() - t, rec(I)))

# %% Step 4 — hnswlib｜HNSW の別の実装。作る時間と検索の速さを比べる。
t = time.time()
h = hnswlib.Index("ip", big.shape[1]); h.init_index(len(big), M=16, ef_construction=80, random_seed=SEED); h.add_items(big, num_threads=1)
b = time.time() - t
h.set_ef(128)
t = time.time(); I, _ = h.knn_query(UV, K)
rows.append(("hnswlib（M=16, ef=128）", b, time.time() - t, rec(I)))

# %% Step 5 — 比較｜作る時間・検索の時間・速さの比は環境で変わるので [time] の行に出し、表には再現率だけを載せる。
base = rows[0][2]
for name, b, s, _ in rows:
    print(f"[time] {name}：作る {b:.1f} 秒・検索 {s:.3f} 秒・総当たりの {base / s:.0f} 倍の速さ")
show(pd.DataFrame([(n, rc) for n, b, s, rc in rows], columns=["方法", f"再現率@{K}（対 総当たり）"]),
     "ANN の比較（24 万件の商品、1,000 人の検索、合成データ）", index=False, digits=3)
