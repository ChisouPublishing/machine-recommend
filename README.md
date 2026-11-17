# 推薦システムのための確率予測モデリング — サンプルコード

書籍のサンプルコードです。ファッションECの合成データで、業務課題・データ設計・モデル比較・選定・運用を学びます。

リポジトリ：https://github.com/ChisouPublishing/machine-recommend

## 取得と環境構築

以下はリポジトリのルートで実行します。確認環境はmacOS（Apple Silicon）、Python 3.12.11、CPUです。ほかのOSでの動作は未確認です。

```bash
git clone https://github.com/ChisouPublishing/machine-recommend.git
cd machine-recommend
uv venv --python 3.12.11 .venv
uv pip install --python .venv/bin/python -r requirements.lock.txt
uv venv --python 3.12.11 .venv-recbole
uv pip install --python .venv-recbole/bin/python -r requirements-ext-recbole.lock.txt
uv venv --python 3.12.11 .venv-ops
uv pip install --python .venv-ops/bin/python -r requirements-ext-ops.lock.txt
./third_party/fetch_fuxictr.sh
.venv/bin/python run_all.py
```

uvの導入と依存パッケージ・FuxiCTRの取得にはネットワーク接続が必要です。RecBoleとFeast・Evidentlyは依存の互換性のため別環境で動かします。

## 構成と再実行

- `00_check/`：環境確認
- `01_business/`〜`13_ops/`：章ごとのスクリプト
- `90_appendix/`：付録
- `common/`：共通処理、モデル、評価・探索
- `data/`：合成データの生成と公開データの取得
- `outputs/`：保存した標準出力・表・図・探索履歴など
- `tests/`：データ・指標・特徴量・探索の検査
- `run_all.py`：環境を切り替えて全章を実行。`--search` で探索をやり直す

```bash
.venv/bin/python -m pytest -q tests
```

仮想環境・生成データ・大きな中間生成物・取得したFuxiCTR本体はGit管理対象外です。生成データはスクリプトで作り直します。保存した探索履歴を使う実行と、探索をやり直す実行では所要時間が異なります。書籍の実測値は確認環境の結果であり、実事業での効果を示すものではありません。

公開データの出典・条件・取得方法は `data/public/README.md` を参照してください。書籍制作側の原稿照合ツール・検証証跡はこのコードリポジトリには含みません。
