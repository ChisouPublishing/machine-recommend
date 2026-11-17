# 公開データの台帳

本書の主データは `data/make_data.py` の合成データ。公開データは、書籍（商用の電子書籍）への結果の掲載と再配布の条件を確かめ、満たすものだけ結果を載せる。確認の証跡は `verification/20261116_public-data/evidence/`（公式ページの本文と画面、2026-11-16 取得）。

| データ | 内容 | 利用条件（公式の記載） | 採否 | 理由と使い方 |
|---|---|---|---|---|
| UCI Online Retail II | 英国のオンライン小売（ギフト雑貨）の 2009-12-01〜2011-12-09 の取引 1,067,371 行 | CC BY 4.0（UCI のデータセットページ） | **採用**（結果を掲載） | 出典を明記すれば商用でも利用できる。実データでの補助検証（共起・候補生成・再購入の予測、第9・11章）に使う。表示ログ・傾向スコアは無い |
| Open Bandit Dataset（ZOZOTOWN） | ファッション EC の推薦枠の表示・クリック・傾向スコア（小サンプルは obp に同梱） | ZOZO Research のページは「大学など公共の研究機関・研究者が研究目的で利用できるよう」公開と記載。データの利用許諾（ライセンス）の明記は見つからない（obp のコードは Apache-2.0） | **不採用**（結果は載せない） | 商用書籍への結果の掲載が許されるか確かめられないため。第4・10章では、読者が自分の環境で試す手順（obp の `OpenBanditDataset`）だけを示す |
| RetailRocket など | EC の行動ログ | CC BY-NC-SA 4.0 など非商用の条件 | 不採用 | 非商用の条件のため |

## UCI Online Retail II

- 出典：Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D （CC BY 4.0）
- 取得：`python data/public/get_online_retail_ii.py`（既定は保存済みを使う。`--download` で取り直す）
- 取得日：2026-11-16。zip の SHA-256：`572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb`
- 変換後（`online_retail_ii.parquet`、2 シートを結合）の SHA-256：`c1667596005866ae538423b24572f98e17fc27214a21fa40c61187e1e2a5277b`
- 列：Invoice、StockCode、Description、Quantity、InvoiceDate、Price、Customer ID、Country。Invoice が C で始まる行は取消（返品）。
