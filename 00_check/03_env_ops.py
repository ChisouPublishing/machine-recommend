"""一括実行の確認：別環境 ext-ops で動く。"""
# env: ext-ops
import pandas, feast, evidently
print("ext-ops", "pandas", pandas.__version__, "feast", feast.__version__)
