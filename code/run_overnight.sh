#!/usr/bin/env bash
# Overnight queue: one job at a time, low CPU priority (nice 15), 2 CPU threads, GPU via MPS.
# Leaves the machine usable. Order: DeepLOB on Nasdaq ITCH, FI-2010 seeds 1-2, FI-2010 z-score (if present).
#
#   nohup caffeinate -i bash code/run_overnight.sh > $HFT_DATA_DIR/logs/overnight.log 2>&1 &
#
# Pause / resume everything (e.g. when you need the GPU):
#   pkill -STOP -f 'train_(deeplob_itch|fi2010).py'      pkill -CONT -f 'train_(deeplob_itch|fi2010).py'
# Stop: pkill -f run_overnight.sh; pkill -f 'train_(deeplob_itch|fi2010).py'
set -u
cd "$(dirname "$0")/.."
export HFT_DATA_DIR="${HFT_DATA_DIR:-/Volumes/SeagateData/hft_data}"
PY="${PY:-/opt/homebrew/opt/python@3.11/libexec/bin/python}"
L="$HFT_DATA_DIR/logs"; mkdir -p "$L"
run() {  # run <log name> <args...>
  local name=$1; shift
  echo "$(date '+%F %T') START $name"
  nice -n 15 "$PY" "$@" --threads 2 > "$L/$name.log" 2>&1
  echo "$(date '+%F %T') END   $name (exit $?): $(grep -E 'TEST' "$L/$name.log" | tail -1)"
}

run itch_deeplob code/models/train_deeplob_itch.py

for seed in 1 2; do
  for k in 10 20 50; do
    run "fi2010_k${k}_seed${seed}" deeplob_reproduction/train_fi2010.py --k "$k" --seed "$seed"
  done
done

# z-score files from the original FI-2010 release, anywhere under $HFT_DATA_DIR/fi2010
# (the release keeps training and test files in separate folders: link them into one)
ZD="$HFT_DATA_DIR/fi2010/zscore"; mkdir -p "$ZD"
find "$HFT_DATA_DIR/fi2010" -path "$ZD" -prune -o -name '*_Dst_NoAuction_ZScore_CF_[789].txt' -print 2>/dev/null \
  | while read -r f; do ln -sf "$f" "$ZD/"; done
if [ -e "$ZD/Train_Dst_NoAuction_ZScore_CF_7.txt" ] && [ -e "$ZD/Test_Dst_NoAuction_ZScore_CF_9.txt" ]; then
  for k in 10 20 50; do
    run "fi2010_k${k}_zscore" deeplob_reproduction/train_fi2010.py --k "$k" --norm ZScore --data-dir "$ZD"
  done
else
  echo "$(date '+%F %T') SKIP  z-score runs: Train_Dst_NoAuction_ZScore_CF_7.txt not found under $HFT_DATA_DIR/fi2010"
fi
echo "$(date '+%F %T') ALL DONE"
