#!/usr/bin/env bash
# Full data pipeline for the Dec 8-12 2025 ITCH week. Safe to re-run; each stage overwrites its outputs.
set -euo pipefail
cd "$(dirname "$0")/.."
export HFT_DATA_DIR="${HFT_DATA_DIR:-/Volumes/SeagateData/hft_data}"
export NUMBA_CACHE_DIR="$HFT_DATA_DIR/.numba_cache"

DAYS=(2025-12-08 2025-12-09 2025-12-10 2025-12-11 2025-12-12)
STEMS=(S120825 S120925 S121025 S121125 S121225)
SYMBOLS=$(python3 -c "import sys; sys.path.insert(0,'code'); import config; print(' '.join(config.SYMBOLS))")

python3 code/data/download_itch.py "${STEMS[@]}"

# Symbol selection: scan all candidates on the first day (~12 min), result fixed in config.SYMBOLS
python3 code/data/itch_parse.py "${DAYS[0]}"
python3 code/data/book.py "${DAYS[0]}"
python3 code/data/select_symbols.py "${DAYS[0]}"

for d in "${DAYS[@]:1}"; do
  python3 code/data/itch_parse.py "$d" --symbols $SYMBOLS   # ~4-10 min per day, disk-bound
  python3 code/data/book.py "$d" --symbols $SYMBOLS
done
python3 code/data/features.py "${DAYS[@]}" --symbols $SYMBOLS
