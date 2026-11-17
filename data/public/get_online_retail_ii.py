"""UCI Online Retail II（CC BY 4.0）を取得し、ハッシュを確かめて Parquet にする。

出典：Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D
既定は保存済みの Parquet を使い、無いときだけ取得する。--download で取り直す。
"""
import argparse, hashlib, io, sys, urllib.request, zipfile
from pathlib import Path
import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
HERE = Path(__file__).resolve().parent
RAW = HERE / "raw" / "online_retail_ii.zip"
OUT = HERE / "online_retail_ii.parquet"
SHA256 = "572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb"  # 2026-11-16 取得（台帳 README.md）


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="公式から取り直す")
    a = ap.parse_args()
    if OUT.exists() and not a.download:
        print(f"保存済みを使います: {OUT.name}")
        return
    RAW.parent.mkdir(parents=True, exist_ok=True)
    if a.download or not RAW.exists():
        try:
            with urllib.request.urlopen(URL, timeout=120) as r:
                RAW.write_bytes(r.read())
        except OSError as e:
            sys.exit(f"取得できませんでした（{e}）。ネットワークを確かめ、{URL} を {RAW} に置いてから再実行してください。")
    h = sha256(RAW)
    print("sha256", h)
    if SHA256 and h != SHA256:
        sys.exit("ハッシュが台帳と違います。公式の版が変わった可能性があるので、上書きせずに止めます。")
    with zipfile.ZipFile(RAW) as z:
        name = [n for n in z.namelist() if n.endswith(".xlsx")][0]
        sheets = pd.read_excel(io.BytesIO(z.read(name)), sheet_name=None,
                               dtype={"Invoice": str, "StockCode": str, "Description": str})
    df = pd.concat(sheets.values(), ignore_index=True)
    df.to_parquet(OUT, index=False)
    print(f"{len(df):,} 行を保存しました: {OUT.name}（シート {list(sheets)}）")


if __name__ == "__main__":
    main()
