"""Shared paths and experiment constants."""
import os
from pathlib import Path

# All market data lives outside OneDrive. Override with HFT_DATA_DIR.
DATA_DIR = Path(os.environ.get("HFT_DATA_DIR", "/Volumes/SeagateData/hft_data")).expanduser()
RAW_DIR = DATA_DIR / "itch_raw"          # S120825-v50.txt.gz ...
EVENTS_DIR = DATA_DIR / "itch_events"    # <day>/<SYM>.parquet  (filtered order events)
BOOK_DIR = DATA_DIR / "book"             # <day>/<SYM>.parquet  (100 ms top-10 snapshots + flow)
FEAT_DIR = DATA_DIR / "features"         # <day>/<SYM>.parquet  (model inputs + labels)

REPO = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO / "results"

# Dec 8-12 2025, consecutive week of free Nasdaq ITCH files.
DAYS = ["2025-12-08", "2025-12-09", "2025-12-10", "2025-12-11", "2025-12-12"]
SPLIT = {"train": DAYS[:3], "val": DAYS[3:4], "test": DAYS[4:]}


def itch_file(day):
    y, m, d = day.split("-")
    return RAW_DIR / f"S{m}{d}{y[2:]}-v50.txt.gz"


# Candidates scanned by code/data/select_symbols.py on 2025-12-08 (order fixes the integer symbol id).
CANDIDATES = ["QQQ", "ADBE", "AMGN", "COST", "ISRG", "VRTX", "INTU", "PEP", "ADP", "KLAC", "TXN", "PANW",
              "AMD", "CSCO", "MDLZ", "SBUX", "GILD", "QCOM", "PYPL", "MU", "AAPL", "MSFT", "NVDA", "AMZN",
              "INTC", "CMCSA", "AMAT", "MAR", "CSX", "KDP", "AEP", "EXC", "XEL", "META", "GOOGL", "AVGO", "TSLA"]
# Selected: QQQ plus the two names with a representative Nasdaq book (1-tick spread >= 50% of the time,
# >= 20k trades/day) and the shortest best-level queue measured in trades. See results/data/.
SYMBOLS = ["QQQ", "PYPL", "NVDA"]

# Snapshot grid
STEP_NS = 100_000_000                    # 100 ms
SESSION_START_NS = (9 * 3600 + 30 * 60) * 1_000_000_000
SESSION_END_NS = 16 * 3600 * 1_000_000_000
TRIM_NS = 5 * 60 * 1_000_000_000         # drop first/last 5 minutes (auction effects)
N_LEVELS = 10
