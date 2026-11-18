"""第7・8章の発展版：FuxiCTR（model_zoo）のモデルを、本書の特徴量の表で学習・予測する包み。

FuxiCTR は pip のパッケージに前処理と学習の仕組みがあり、モデルの実装はリポジトリの model_zoo にある（第2章の互換確認）。
third_party/fetch_fuxictr.sh で取得した版（v2.3.10）を使う。
"""
# %% Step 1 — 準備｜model_zoo を読み込めるようにし、列の仕様（カテゴリ・数値・系列）を FuxiCTR の形で書く。
from __future__ import annotations

import logging
import os
import shutil
import sys

import numpy as np
import pandas as pd
import yaml

from common.features import NUM_COLS
from common.models.torch_ctr import FIELDS, SEQ_LEN
from common.paths import CODE, OUT

FX = CODE / "third_party" / "FuxiCTR"
if not FX.exists():
    raise SystemExit("FuxiCTR がありません。third_party/fetch_fuxictr.sh を実行してください")
sys.path.insert(0, str(FX))
logging.getLogger().setLevel(logging.WARNING)

from fuxictr.features import FeatureMap  # noqa: E402
from fuxictr.preprocess import FeatureProcessor, build_dataset  # noqa: E402
from fuxictr.pytorch.dataloaders import RankDataLoader  # noqa: E402
from fuxictr.pytorch.torch_utils import seed_everything  # noqa: E402
import model_zoo  # noqa: E402
import ray  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)   # 1 スレッドで学習する（結果を揺らさない。LightGBM の OpenMP とぶつかって落ちるのも防ぐ。付録 A）

# Ray は、ディスクの使用率が 95% を超えると一時ファイルを作らずに止まる。本書の表は数百 MB なので、空きの少ない環境でも
# 動くよう、止める閾値を 99% に上げる（2026-11-18、第8章 06 で確認）
os.environ.setdefault("RAY_local_fs_capacity_threshold", "0.99")
if not ray.is_initialized():
    ray.init(ignore_reinit_error=True, logging_level=logging.ERROR, num_cpus=1, include_dashboard=False)

# 数値の列は、学習期間の分位点で 20 区間に分けてカテゴリにする（FuxiCTR は数値の列の埋め込みを Xavier 正規分布で
# 初期化するため、数値の列が多いと学習の始めの予測が極端になる。区間に分けるのは CTR 予測の定番の前処理）
BINS = 20
BASE_COLS = [{"name": FIELDS + [f"{c}_bin" for c in NUM_COLS], "active": True, "dtype": "str", "type": "categorical"}]
# 閲覧の系列は、系列を注意機構で使うモデル（DIN・DIEN）にだけ渡す。FM 系の 1 次の項は系列を和でまとめる作りで、
# 20 件分の和が学習を不安定にするため（最小実装の FM・DeepFM も系列を使わないので、入力がそろう）
SEQ_COLS = [{"name": "hist_items", "active": True, "dtype": "str", "type": "sequence", "splitter": "^", "max_len": SEQ_LEN,
             "share_embedding": "item_id", "padding": "pre"},
            {"name": "hist_cats", "active": True, "dtype": "str", "type": "sequence", "splitter": "^", "max_len": SEQ_LEN,
             "share_embedding": "category_id", "padding": "pre"}]
SUB = {"DeepFM": "DeepFM/DeepFM_torch", "WideDeep": "WideDeep/WideDeep_torch", "DNN": "DNN/DNN_torch",
       "ShareBottom": "multitask/ShareBottom", "MMoE": "multitask/MMoE", "PLE": "multitask/PLE"}   # 多タスク（第8章 8.6）


# %% Step 2 — データを渡す｜学習・早期終了用・評価の 3 つを CSV に書き、FuxiCTR の前処理で語彙と Parquet を作る（語彙と標準化は学習だけから）。
def prepare(tag: str, train: pd.DataFrame, holdout: pd.DataFrame, test: pd.DataFrame, label="purchased",
            with_seq: bool = False, numeric: str = "bin"):
    """numeric="bin"：数値を学習期間の分位点で区間に分ける（既定）。"raw"：数値のまま渡し FuxiCTR に標準化させる（落とし穴の再現用）。
    label にリスト（例：["clicked", "purchased"]）を渡すと、多タスクのモデル（第8章 8.6）用に複数の正解を持たせる。"""
    labels = [label] if isinstance(label, str) else list(label)
    root = OUT / "fuxictr"
    d = root / tag
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    edges = {c: np.unique(np.quantile(train[c], np.linspace(0, 1, BINS + 1)[1:-1])) for c in NUM_COLS}   # 学習期間の分位点
    paths = {}
    train = train.sample(frac=1.0, random_state=0)                      # 学習の行の順は、ここで種を決めて並べ替える（2 の Step 3 で説明）
    for name, part in (("train", train), ("valid", holdout), ("test", test)):
        paths[name] = str(d / f"{name}.csv")
        out = part[FIELDS + ["hist_items", "hist_cats"]].copy()
        for c in NUM_COLS:
            if numeric == "bin":
                out[f"{c}_bin"] = np.searchsorted(edges[c], part[c].to_numpy(), side="right")
            else:
                out[c] = np.log1p(np.clip(part[c].to_numpy(float), 0, None))
        for lb in labels:
            out[lb] = part[lb].astype(float).to_numpy()
        out.to_csv(paths[name], index=False)
    base = BASE_COLS if numeric == "bin" else [{"name": FIELDS, "active": True, "dtype": "str", "type": "categorical"},
                                               {"name": NUM_COLS, "active": True, "dtype": "float", "type": "numeric",
                                                "normalizer": "StandardScaler"}]
    fp = FeatureProcessor(feature_cols=base + (SEQ_COLS if with_seq else []), label_col=[{"name": lb, "dtype": "float"} for lb in labels] if len(labels) > 1 else {"name": labels[0], "dtype": "float"}, dataset_id=tag,
                          data_root=str(root), min_categr_count=1)
    tr, va, te = build_dataset(fp, train_data=paths["train"], valid_data=paths["valid"], test_data=paths["test"])
    for p_ in paths.values():                                            # 変換に使った CSV は消す（Parquet が残る。ディスクを節約）
        os.remove(p_)
    return {"tag": tag, "root": str(root), "train": tr, "valid": va, "test": te}


# %% Step 3 — 学習と予測｜model_zoo の設定ファイルの既定値に、本書の設定（埋め込みの次元・学習率・エポックなど）を上書きして学習する。
def fit_predict(model_name: str, data: dict, overrides: dict, seed: int = 0, return_model: bool = False,
                sample_batch: int = 0):
    """予測（data["test"] の行の順）を返す。return_model=True なら (予測, モデル, 評価の先頭 sample_batch 行の入力) を返す（速さの計測用）。"""
    sub = SUB.get(model_name, model_name)
    cfg = yaml.safe_load(open(FX / "model_zoo" / sub / "config" / "model_config.yaml"))
    key = f"{model_name}_default" if f"{model_name}_default" in cfg else f"{model_name}_test"
    p = {**cfg["Base"], **cfg[key]}
    p.update(dataset_id=data["tag"], data_root=data["root"], data_format="parquet", train_data=data["train"],
             valid_data=data["valid"], test_data=data["test"], model_root=str(OUT / "fuxictr" / "ckpt"), gpu=-1,
             num_workers=0, verbose=0, seed=seed, batch_size=4096, epochs=3, early_stop_patience=1,
             monitor="logloss", monitor_mode="min", metrics=["logloss", "AUC"], feature_specs=None)
    if model_name in ("DIN", "DIEN"):
        f = model_name.lower()
        p.update({f"{f}_target_field": ["item_id", "category_id"], f"{f}_sequence_field": ["hist_items", "hist_cats"],
                  "feature_specs": [{"name": "hist_items", "feature_encoder": None},
                                    {"name": "hist_cats", "feature_encoder": None}]})
        if model_name == "DIEN":
            p.update(dien_neg_seq_field=[])
    p.update(overrides)
    os.makedirs(os.path.join(p["model_root"], data["tag"]), exist_ok=True)
    seed_everything(seed)
    fmap = FeatureMap(data["tag"], os.path.join(data["root"], data["tag"]))
    fmap.load(os.path.join(data["root"], data["tag"], "feature_map.json"), p)
    # FuxiCTR の読み込み（Ray Data）は、並べ替えに乱数の種を渡せず、結果が実行ごとに変わる。そこで並べ替えを切り、
    # 行の順を保つ設定にする（学習の行は prepare で種を決めて並べ替え済み。エポックごとの並べ替えはしない）
    ray.data.DataContext.get_current().execution_options.preserve_order = True
    tr, va = RankDataLoader(fmap, stage="train", train_data=p["train_data"], valid_data=p["valid_data"],
                            batch_size=p["batch_size"], data_format="parquet", shuffle=False)
    model = getattr(model_zoo, model_name)(fmap, **p)
    model.fit(tr, validation_data=va, epochs=p["epochs"])                # 早期終了は学習期間の末の 7 日（holdout）で判断する
    te = RankDataLoader(fmap, stage="test", test_data=p["test_data"], batch_size=p["batch_size"], data_format="parquet",
                        shuffle=False)
    pred = model.predict(te)
    pred = ({k: np.asarray(v, dtype=np.float64) for k, v in pred.items()} if isinstance(pred, dict)   # 多タスク：正解の列名ごと
            else np.asarray(pred, dtype=np.float64))
    if not return_model:
        return pred
    one = RankDataLoader(fmap, stage="test", test_data=p["test_data"], batch_size=sample_batch or 300, data_format="parquet",
                         shuffle=False)
    batch = next(iter(one))
    return pred, model, batch
