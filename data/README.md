# Data

Raw and processed market data are **not** committed (licensing and size). Everything lives under
`$HFT_DATA_DIR`, which defaults to `/Volumes/SeagateData/hft_data` (see `code/config.py`). Keep it outside
OneDrive: one raw day is ~9 GB compressed.

```bash
export HFT_DATA_DIR=/Volumes/SeagateData/hft_data   # add to ~/.zshrc
```

## Source

Free Nasdaq TotalView-ITCH 5.0 sample files, https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/ (public, no login,
no licence terms posted; used for research only). Each file is one full trading day of every add, cancel,
replace and execution on Nasdaq, with order IDs and nanosecond timestamps (true L3).

| Use | Days (file stem) | Size |
|---|---|---|
| Train | 2025-12-08 (S120825), 2025-12-09 (S120925), 2025-12-10 (S121025) | ~27 GB |
| Validation | 2025-12-11 (S121125) | ~9 GB |
| Test | 2025-12-12 (S121225) | ~9 GB |
| Extra out-of-time test (optional) | 2026-05-15, 2026-05-18, 2026-06-12 | ~47 GB |

Symbols: QQQ plus two mid-liquidity Nasdaq stocks picked by `code/data/select_symbols.py`
(table saved to `results/data/`).

## Layout and commands

```
$HFT_DATA_DIR/
  itch_raw/S120825-v50.txt.gz        python code/data/download_itch.py S120825 S120925 S121025 S121125 S121225
  itch_events/<day>/<SYM>.parquet    python code/data/itch_parse.py 2025-12-08           (~4-12 min/day)
  book/<day>/<SYM>.parquet           python code/data/book.py 2025-12-08 ...             (100 ms top-10 book + flow)
  features/<day>/<SYM>.parquet       python code/data/features.py 2025-12-08 ... --symbols QQQ X Y
```

`code/run_pipeline.sh` runs all stages for the five days.

## Benchmark dataset

FI-2010 (Ntakaris et al. 2018) for the DeepLOB reproduction, in the DeepLOB authors' preprocessed form
(decimal-precision normalisation, no auction; train = days 1-7, test = days 8-10 as in the paper):

```bash
cd $HFT_DATA_DIR/fi2010
curl -LO https://raw.githubusercontent.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/master/data/data.zip
unzip data.zip   # Train_Dst_NoAuction_DecPre_CF_7.txt, Test_Dst_NoAuction_DecPre_CF_{7,8,9}.txt
```
