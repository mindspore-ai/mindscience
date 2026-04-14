#!/bin/bash
set -euo pipefail

src_dir="${1:-./case/}"
dst_dir="${2:-user@login:/path/to/project/case/}"

rsync -avh -e ssh --partial --progress "$src_dir" "$dst_dir"
