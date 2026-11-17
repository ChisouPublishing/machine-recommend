"""第3章：時点の検査が、正しい特徴量では通り、未来の記録を混ぜた特徴量では失敗することを確かめる。"""
import pytest

import make_data as m
from common import features
from common.data import connect
from common.leakage import audit_sequence, audit_user_views, D


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp("small")
    for name, df in m.generate(m.Params(scale=0.03)).items():
        df.to_parquet(d / f"{name}.parquet", index=False)
    con = connect(data_dir=d)
    return features.build(con), con


def test_correct_features_pass(built):
    df, con = built
    assert len(audit_user_views(df, con)) == 0
    assert len(audit_user_views(df, con, col="u_views_7d", window=7)) == 0
    assert len(audit_sequence(df, con)) == 0


def test_leaky_feature_detected(built):
    df, con = built
    # 誤った書き方：表示の日「当日」の閲覧まで数える（表示の後のクリックも入る）
    con.register("x", df[["impression_id", "user_id", "day"]])
    leaky = con.sql(f"""SELECT x.impression_id, COUNT(e.event_id) AS u_views_28d FROM x
        LEFT JOIN event e ON e.user_id = x.user_id AND e.event_type = 'view'
         AND {D.format('e.occurred_at')} <= x.day AND {D.format('e.occurred_at')} >= x.day - 28 GROUP BY 1""").df()
    con.unregister("x")
    bad = df.drop(columns="u_views_28d").merge(leaky, on="impression_id")
    assert len(audit_user_views(bad, con)) > 0


def test_leaky_sequence_detected(built):
    df, con = built
    bad = df.copy()
    bad["hist_items"] = bad["hist_items"] + "^" + bad["item_id"].astype(str)   # 表示した商品（未来の閲覧）を系列に足す
    assert len(audit_sequence(bad, con)) > 0
