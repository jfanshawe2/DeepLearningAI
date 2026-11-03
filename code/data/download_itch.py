"""Download free Nasdaq TotalView-ITCH 5.0 sample days from emi.nasdaq.com.

The server caps each connection at ~0.8 MB/s, so each file is fetched as many byte ranges in parallel
and written in place. Finished chunks are recorded in <file>.parts, so an interrupted run resumes.

    python code/data/download_itch.py S120825 S120925 S121025 S121125 S121225 --workers 16
"""
import argparse
import json
import os
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/"
CHUNK = 64 * 1024 * 1024
UA = {"User-Agent": "itch-research-downloader"}


def size_of(url):
    req = urllib.request.Request(url, method="HEAD", headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def fetch(url, path, start, end, retries=5):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r, open(path, "r+b") as f:
                f.seek(start)
                pos = start
                while True:
                    buf = r.read(1 << 20)
                    if not buf:
                        break
                    f.write(buf)
                    pos += len(buf)
            if pos == end + 1:
                return start
            raise IOError(f"short read {pos - start} of {end - start + 1}")
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            print(f"  retry {start}: {e}", flush=True)


def download(name, out_dir, workers):
    fname = f"{name}-v50.txt.gz"
    url, path = BASE + fname, out_dir / fname
    parts_path = path.with_suffix(path.suffix + ".parts")
    total = size_of(url)
    if path.exists() and not parts_path.exists() and path.stat().st_size == total:
        print(f"{fname}: already complete")
        return
    if not path.exists():
        with open(path, "wb") as f:
            f.truncate(total)
    done = set(json.loads(parts_path.read_text())) if parts_path.exists() else set()
    todo = [s for s in range(0, total, CHUNK) if s not in done]
    print(f"{fname}: {total / 1e9:.2f} GB, {len(todo)} chunks to fetch", flush=True)
    lock = threading.Lock()
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(fetch, url, path, s, min(s + CHUNK, total) - 1) for s in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            with lock:
                done.add(fut.result())
                parts_path.write_text(json.dumps(sorted(done)))
            if i % 10 == 0 or i == len(futs):
                print(f"  {fname}: {len(done) * CHUNK / total:.0%}", flush=True)
    parts_path.unlink()
    print(f"{fname}: complete", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("days", nargs="+", help="file stems, e.g. S120825")
    p.add_argument("--out", default=os.path.join(os.environ.get("HFT_DATA_DIR", "~/hft_data"), "itch_raw"))
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    for d in args.days:
        download(d, out, args.workers)


if __name__ == "__main__":
    main()
