"""第13章：発展版の特徴量ストア（Feast）。01 と同じ日次の集計を Feast に登録し、学習用（時点指定）と推論用（オンライン）の値が 01 と一致するかを確かめる。"""
# env: ext-ops
# %% Step 1 — リポジトリの設定｜オフラインはファイル（Parquet）、オンラインは SQLite（ローカルで動く最小の構成）。
import contextlib
import io
import shutil
import textwrap
from datetime import datetime, timedelta

import pandas as pd
from feast import Entity, FeatureStore, FeatureView, Field, FileSource, ValueType
from feast.types import Int64

from common import OUT

src = OUT / "13_ops" / "store" / "user_views.parquet"
repo = OUT / "13_ops" / "feast_repo"
shutil.rmtree(repo, ignore_errors=True)
repo.mkdir(parents=True)
start = datetime(2025, 4, 7)
df = pd.read_parquet(src)
df["event_timestamp"] = [start + timedelta(days=int(d)) for d in df.as_of]   # その日の 0 時の値
df.drop(columns="as_of").to_parquet(repo / "user_views.parquet", index=False)
(repo / "feature_store.yaml").write_text(textwrap.dedent(f"""
    project: suisen
    registry: {repo}/registry.db
    provider: local
    online_store: {{type: sqlite, path: {repo}/online.db}}
    offline_store: {{type: file}}
    entity_key_serialization_version: 3
    """))

# %% Step 2 — 登録｜会員（エンティティ）と、閲覧数の FeatureView。ttl は 1 日（前日以前の値は使わない＝01 の as_of と同じ約束）。
user = Entity(name="user", join_keys=["user_id"], value_type=ValueType.INT64)
fv = FeatureView(name="user_views", entities=[user], ttl=timedelta(days=1),
                 schema=[Field(name="u_views_7d", dtype=Int64), Field(name="u_views_28d", dtype=Int64)],
                 source=FileSource(path=str(repo / "user_views.parquet"), timestamp_field="event_timestamp"))
store = FeatureStore(repo_path=str(repo))
store.apply([user, fv])

# %% Step 3 — 学習用｜(会員, 時刻) の一覧で、その時刻より前の最新の値を取る（get_historical_features）。01 の値と比べる。
ref = df[df.as_of == 140].sort_values("user_id").head(1000)
ent = pd.DataFrame({"user_id": ref.user_id.to_numpy(), "event_timestamp": start + timedelta(days=140, hours=12)})
hist = store.get_historical_features(entity_df=ent, features=["user_views:u_views_7d", "user_views:u_views_28d"]).to_df()
hist = hist.sort_values("user_id").reset_index(drop=True)
print(f"学習用：01 との食い違い {int((hist.u_views_28d.to_numpy() != ref.u_views_28d.to_numpy()).sum())} 件 / {len(ref):,} 人")

# %% Step 4 — 推論用｜日 140 までの値をオンラインに書き込み（materialize）、get_online_features で 1 人ずつ取り出す。
with contextlib.redirect_stdout(io.StringIO()):                          # 進み具合の表示（色の制御文字を含む）を出さない
    store.materialize(start + timedelta(days=139), start + timedelta(days=140, hours=1))
onl = store.get_online_features(features=["user_views:u_views_7d", "user_views:u_views_28d"],
                                entity_rows=[{"user_id": int(u)} for u in ref.user_id]).to_df()
print(f"推論用：01 との食い違い {int((onl.u_views_28d.to_numpy() != ref.u_views_28d.to_numpy()).sum())} 件 / {len(ref):,} 人")
print(onl.head(3).to_string(index=False))
