"""本書のサンプルコードの共通処理（データの読み込み・期間の分割・指標・出力の保存）。"""
# macOS では、PyTorch を LightGBM より先に読み込むと、OpenMP の競合で落ちる（2026-11-17 に確認）。common を先に読めば
# LightGBM が先になるよう、ここで読み込んでおく
try:
    import lightgbm  # noqa: F401
except ImportError:                       # 別環境（RecBole・Feast）には LightGBM が無い
    pass
from .paths import CODE, DATA, OUT, SEED, out_dir
from .data import connect, load, periods, cv_folds, PERIODS
from .metrics import auc, logloss, pr_auc, ece, ndcg_at_k, recall_at_k, bootstrap_ci, summary
from .report import show, save_table, savefig, timer

__all__ = ["CODE", "DATA", "OUT", "SEED", "out_dir", "connect", "load", "periods", "cv_folds", "PERIODS",
           "auc", "logloss", "pr_auc", "ece", "ndcg_at_k", "recall_at_k", "bootstrap_ci", "summary",
           "show", "save_table", "savefig", "timer"]
