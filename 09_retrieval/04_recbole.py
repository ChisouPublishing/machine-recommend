"""第9章：RecBole で、系列推薦（SASRec）とグラフ（LightGCN）の候補を作り、同じ評価で比べる。"""
# env: ext-recbole
# %% Step 1 — データ｜学習 day 77〜132（直近 8 週）の行動（閲覧・カート・購入、直帰を除く）を、RecBole の形式（.inter）に書き出す。
#               RecBole の学習は CPU では重いので、期間を絞り、候補は保存して既定では読み込む（作り直しは run_all.py --search）。
import common  # noqa: F401  先に読み込む（LightGBM を PyTorch より先に。付録 A）
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import LightGCN
from recbole.model.sequential_recommender import SASRec
from recbole.trainer import Trainer
from recbole.utils import init_seed
from recbole.utils.case_study import full_sort_topk

from common import SEED, connect, show
from common.retrieval import TRAIN, eval_users, recall, true_recall, true_top

torch.set_num_threads(1)   # 並列の足し合わせの順で結果が揺れないよう、1 スレッドで学習する
con = connect()
D = "date_diff('day', DATE '2025-04-07', CAST({} AS DATE))"
ev = con.sql(f"""SELECT user_id, item_id, epoch(occurred_at) AS ts FROM event
 WHERE event_type IN ('view', 'cart', 'purchase') AND source <> 'landing' AND {D.format('occurred_at')} BETWEEN 77 AND {TRAIN[1]}
 ORDER BY ts, user_id, item_id""").df()
root = os.path.abspath("outputs/09_retrieval/recbole")
os.makedirs(f"{root}/suisen", exist_ok=True)
ev.rename(columns={"user_id": "user_id:token", "item_id": "item_id:token", "ts": "timestamp:float"}).to_csv(
    f"{root}/suisen/suisen.inter", sep="\t", index=False)
users, truth = eval_users()
tt = true_top()
need = np.union1d(users, tt.user_id.unique())
rows = []
cache = os.path.abspath("outputs/09_retrieval/recbole_cands.pkl")
saved = pickle.load(open(cache, "rb")) if os.path.exists(cache) and os.environ.get("SUISEN_SEARCH") != "1" else {}

# %% Step 2 — 学習｜各会員の最後の 2 件は RecBole の検証・テストに分かれる（時系列の順）が、毎エポックの検証は省く。
#               SASRec は直近 50 件の系列（3 エポック）、LightGCN は会員×商品の 2 部グラフ（30 エポック）。
for name, M, extra in (("SASRec（RecBole）", SASRec, {"epochs": 3, "MAX_ITEM_LIST_LENGTH": 50, "train_neg_sample_args": None,
                                                     "loss_type": "CE", "train_batch_size": 2048}),
                       ("LightGCN（RecBole）", LightGCN, {"epochs": 30, "train_batch_size": 8192})):
    if name in saved:
        cands = saved[name]
        rows.append({"方法": name, **recall({u: cands.get(u, []) for u in users}, truth), **true_recall(cands, tt),
                     "候補のある会員": len(cands)})
        continue
    cfg = Config(model=M.__name__, dataset="suisen", config_dict={
        "data_path": root, "load_col": {"inter": ["user_id", "item_id", "timestamp"]}, "seed": SEED, "reproducibility": True,
        "eval_args": {"split": {"LS": "valid_and_test"}, "order": "TO", "mode": "full"}, "device": "cpu", "use_gpu": False,
        "show_progress": False, "state": "ERROR", "checkpoint_dir": f"{root}/ckpt", "valid_metric": "Recall@50",
        "metrics": ["Recall"], "topk": [50], "eval_batch_size": 4096, **extra})
    init_seed(cfg["seed"], cfg["reproducibility"])
    ds = create_dataset(cfg)
    tr_d, va_d, te_d = data_preparation(cfg, ds)
    model = M(cfg, tr_d.dataset).to("cpu")
    t = time.time()
    Trainer(cfg, model).fit(tr_d, None, saved=False, show_progress=False)   # 内部の検証は省く（評価は本書の方法で行う）
    print(f"[time] {name} の学習 {time.time() - t:.1f} 秒")

    # %% Step 3 — 候補｜評価の会員ごとに、全商品の点数の上位 100 件（RecBole の full_sort_topk）。学習に無い会員は候補なし。
    known = [u for u in need if str(u) in ds.field2token_id["user_id"]]
    uid = ds.token2id(ds.uid_field, [str(u) for u in known])
    ok = np.array([te_d.uid2history_item[i] is not None for i in uid])   # RecBole のテストの分割に入った会員だけ採点できる
    known, uid = list(np.array(known)[ok]), uid[ok]
    cands = {}
    for s in range(0, len(uid), 1024):
        _, top = full_sort_topk(uid[s:s + 1024], model, te_d, k=100, device="cpu")
        for u, r in zip(known[s:s + 1024], top.numpy()):
            cands[u] = [int(x) for x in ds.id2token(ds.iid_field, r)]
    rows.append({"方法": name, **recall({u: cands.get(u, []) for u in users}, truth), **true_recall(cands, tt),
                 "候補のある会員": len(cands)})
    saved[name] = cands
    pickle.dump(saved, open(cache, "wb"))

# %% Step 4 — 比較｜01（人気・共起・ALS・BPR）と同じ評価の表に並べる。
show(pd.DataFrame(rows).set_index("方法"), "RecBole の候補（評価 day 133〜146、合成データ）")
