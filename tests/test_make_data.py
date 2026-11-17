"""合成データ生成器の再現性・整合性・性質の検査（小さな規模で生成する）。"""
import numpy as np
import pandas as pd
import pytest

import make_data as m


@pytest.fixture(scope="module")
def tabs():
    return m.generate(m.Params(scale=0.03))


def test_reproducible(tabs):
    again = m.generate(m.Params(scale=0.03))
    for name, df in tabs.items():
        assert m.table_hash(df) == m.table_hash(again[name]), name


def test_seed_changes_data(tabs):
    other = m.generate(m.Params(scale=0.03, seed=1))
    assert m.table_hash(tabs["impression"]) != m.table_hash(other["impression"])


def test_propensity_positive(tabs):
    p = tabs["impression"].propensity
    assert (p > 0).all() and (p <= 1).all()


def test_foreign_keys(tabs):
    t = tabs
    def sub(child, col, parent, pcol=None):
        v = t[child][col].dropna().astype("int64")
        assert v.isin(t[parent][pcol or col]).all(), f"{child}.{col} -> {parent}"
    sub("item", "category_id", "category"); sub("item", "brand_id", "brand"); sub("sku", "item_id", "item")
    sub("session", "user_id", "user"); sub("impression", "user_id", "user"); sub("impression", "item_id", "item")
    sub("impression", "session_id", "session"); sub("event", "session_id", "session"); sub("event", "item_id", "item")
    sub("event", "impression_id", "impression"); sub("order", "user_id", "user"); sub("order_line", "order_id", "order")
    sub("order_line", "sku_id", "sku"); sub("return", "order_line_id", "order_line")
    sub("campaign_assignment", "campaign_id", "campaign"); sub("campaign_assignment", "user_id", "user")
    sub("inventory", "sku_id", "sku"); sub("item_availability", "item_id", "item")
    assert t["truth_impression"].impression_id.equals(t["impression"].impression_id)


def test_sku_matches_item(tabs):
    ol = tabs["order_line"].merge(tabs["sku"][["sku_id", "item_id"]], on="sku_id", suffixes=("", "_sku"))
    assert (ol.item_id == ol.item_id_sku).all()


def test_no_truth_columns_in_logs(tabs):
    for name, df in tabs.items():
        if not name.startswith("truth_"):
            assert not any(c.startswith("p_") or c in ("p0", "p1", "uplift") for c in df.columns), name


def test_time_order(tabs):
    ev = tabs["event"].dropna(subset=["impression_id"])
    shown = tabs["impression"].set_index("impression_id").shown_at
    assert (ev.occurred_at.values >= shown.loc[ev.impression_id.astype("int64")].values).all()
    assert tabs["event"].occurred_at.is_monotonic_increasing


def test_realistic_ranges(tabs):
    t = tabs
    clicked = t["event"].query("event_type == 'view'").impression_id.dropna().astype("int64")
    top = t["impression"].query("surface == 'top'")
    ctr = top.impression_id.isin(set(clicked)).mean()
    assert 0.01 < ctr < 0.10
    rr = len(t["return"]) / len(t["order_line"])
    assert 0.05 < rr < 0.25
    assert (t["user"].registered_at >= m.START).mean() > 0.15          # 期間中の新規会員
    assert (t["item"].release_date >= m.START).mean() > 0.30            # 期間中の新商品
    ca = t["campaign_assignment"]
    assert abs(ca.assignment_prob.unique()[0] - 0.5) < 1e-9 and 0.4 < (ca.group == "treatment").mean() < 0.6
    tc = t["truth_campaign"]
    assert tc.uplift.mean() > 0 and (tc.uplift < 0).mean() > 0.05      # 効果が負の層もある


def test_truth_recomputes_logged_p_click(tmp_path):
    p = m.Params(scale=0.03)
    t = m.generate(p)
    for name, df in t.items():
        df.to_parquet(tmp_path / f"{name}.parquet", index=False)
    import truth
    tr = truth.Truth(tmp_path)
    imp = t["impression"].query("surface == 'top'")
    imp = imp[imp.session_id.astype("Int64").isin(tr.state.index)].copy()
    assert len(imp) > 100
    imp["session_id"] = imp.session_id.astype("int64")
    got = tr.p_click(imp.session_id.values, imp.item_id.values[:, None], imp.position.values[:, None]).ravel()
    want = imp.merge(t["truth_impression"], on="impression_id").p_click.values
    np.testing.assert_allclose(got, want, rtol=1e-6)


def test_logged_items_were_available(tabs):
    """表示した商品は、その日に出せる商品だった（深夜をまたいだセッションは前日の一覧で決まる）。"""
    av = set(zip(tabs["item_availability"].item_id, tabs["item_availability"].date.dt.normalize()))
    imp = tabs["impression"]
    d = imp.shown_at.dt.normalize()
    ok = [(i, a) in av or (i, a - pd.Timedelta(days=1)) in av for i, a in zip(imp.item_id, d)]
    assert all(ok)
