#!/usr/bin/env bash
# FuxiCTR の model_zoo（モデルの実装。pip の fuxictr には入っていない）を、版を固定して取得する。
# 取得先：third_party/FuxiCTR（Apache-2.0。git 管理外）
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAG="v2.3.10"
COMMIT="cd393c506e4c066d8f8b0c225230b119b28527dc"
if [[ ! -d "$HERE/FuxiCTR/.git" ]]; then
    git clone -q --depth 1 --branch "$TAG" https://github.com/reczoo/FuxiCTR.git "$HERE/FuxiCTR"
fi
GOT="$(git -C "$HERE/FuxiCTR" rev-parse HEAD)"
if [[ "$GOT" != "$COMMIT" ]]; then
    echo "FuxiCTR のコミットが想定と違います: $GOT（想定 $COMMIT）" >&2
    exit 1
fi
echo "FuxiCTR $TAG ($COMMIT) を $HERE/FuxiCTR に用意しました"
