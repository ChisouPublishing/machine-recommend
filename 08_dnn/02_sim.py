"""第8章：SIM（縮小版）。直近 20 件だけを見る DIN に、長い系列（100 件）からの同じカテゴリの検索を足すと変わるかを確かめる。"""
# %% Step 1 — 長い系列を作る｜セッションの開始より前の、直近 100 件の閲覧の商品とカテゴリ（第3章の系列と同じ時点の約束）。
import common  # noqa: F401  先に読み込む（LightGBM を PyTorch より先に。付録 A）
import pandas as pd
import torch

from common import SEED, connect, show, summary
from common.features import NUM_COLS, load_top, split
from common.models.gbdt import correct, downsample
from common.models.torch_ctr import DIN, LONG_LEN, SIM, Encoder, train

torch.set_num_threads(1)   # 並列の足し合わせの順で結果が揺れないよう、1 スレッドで学習する
R = 0.1
df = load_top()
tr = downsample(split(df, 28, 132), "purchased", R, seed=SEED)
va = split(df, 133, 146)
con = connect()
con.register("s", pd.concat([tr, va])[["session_id", "user_id"]].drop_duplicates())
long = con.sql(f"""
WITH v AS (SELECT s.session_id, e.item_id, it.category_id,
                  ROW_NUMBER() OVER (PARTITION BY s.session_id ORDER BY e.occurred_at DESC, e.event_id DESC) AS rn
           FROM s JOIN session ss USING (session_id)
           JOIN event e ON e.user_id = s.user_id AND e.event_type = 'view' AND e.occurred_at < ss.started_at
           JOIN item it ON it.item_id = e.item_id)
SELECT session_id, string_agg(item_id::VARCHAR, '^' ORDER BY rn DESC) AS hist_items_long,
       string_agg(category_id::VARCHAR, '^' ORDER BY rn DESC) AS hist_cats_long
FROM v WHERE rn <= {LONG_LEN} GROUP BY 1""").df()
tr, va = (d.merge(long, on="session_id", how="left").fillna({"hist_items_long": "", "hist_cats_long": ""}) for d in (tr, va))
n = va.hist_items_long.str.count(r"\^") + (va.hist_items_long != "")
print(f"評価の行の長い系列の長さ：中央値 {n.median():.0f}・90% 点 {n.quantile(0.9):.0f}（上限 {LONG_LEN}）")

# %% Step 2 — 学習と比較｜同じ設定（埋め込み 8 次元・3 エポック）で、DIN と SIM（縮小版）を種 3 つずつ学習する。
enc = Encoder().fit(tr)
xtr, xva = enc.transform(tr), enc.transform(va)
y = va.purchased.to_numpy()
rows = {}
for name, M in (("DIN（直近 20 件）", DIN), ("SIM（直近 20 件＋同じカテゴリの検索 100 件）", SIM)):
    for s in range(3):
        m = M(enc.sizes(), len(NUM_COLS), k=8)
        rows[(name, s)] = summary(y, correct(train(m, xtr, tr.purchased.to_numpy(), xva, epochs=3, lr=2e-3, seed=s), R))
r = pd.DataFrame(rows).T
show(r.groupby(level=0).agg(["mean", "std"])[["AUC", "LogLoss"]], "DIN と SIM（縮小版）（評価 day 133〜146、種 3 つの平均と標準偏差、合成データ）", digits=4)
