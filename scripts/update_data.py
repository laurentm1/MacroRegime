#!/usr/bin/env python3
"""
scripts/update_data.py — Single FRED data refresh script
=========================================================
This is the ONLY file that requires fredapi and a FRED API key.
Run it to pull fresh data from FRED and write data/fred_raw.parquet.

After running this script, the classifier and dashboard work fully
offline — no fredapi, no network, no API key needed by them.

Usage
-----
    export FRED_API_KEY=your_key_here
    python scripts/update_data.py

    # Optional flags:
    python scripts/update_data.py --start 2000-01-01      # default
    python scripts/update_data.py --refresh-history       # force full re-download
    python scripts/update_data.py --out-dir data          # default output dir

Outputs
-------
    data/fred_raw.parquet        wide DataFrame, one column per FRED series
                                 (19 columns), DatetimeIndex from --start to today
    regime_history.csv           re-generated monthly regime labels 2000–today
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure repo root is on path so we can import loaders
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from loaders import FredLoader, SERIES, PARQUET_PATH
from regime_classifier import RegimeClassifier


def fetch_and_save(api_key: str, start: str, out_path: Path):
    print("=" * 60)
    print("  MACRO REGIME SYSTEM — Data Refresh")
    print("=" * 60)
    print(f"  Start date : {start}")
    print(f"  Output     : {out_path}")
    print()

    loader = FredLoader(api_key=api_key, start=start)

    frames = {}
    for name in SERIES:
        try:
            s = loader.get(name)
            print(f"  ✓ {name:20s}  {len(s):5d} obs  "
                  f"latest: {s.index[-1].date() if len(s) else 'N/A'}")
            frames[name] = s
        except Exception as e:
            print(f"  ✗ {name:20s}  ERROR: {e}")

    # Build wide DataFrame on a daily index; fill forward for lower-frequency series
    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Persist
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, engine="pyarrow", compression="snappy")
    print(f"\n✓ Saved {out_path}  ({out_path.stat().st_size // 1024} KB, "
          f"{len(df)} rows × {len(df.columns)} cols)")

    return loader


def refresh_regime_history(loader, start: str, out_path: Path):
    print("\nRegenerating regime_history.csv ...")
    classifier = RegimeClassifier(loader=loader)

    dates = pd.date_range(start=start, end=pd.Timestamp.today(), freq="MS")
    records = []
    for dt in dates:
        try:
            result = classifier.classify(dt)
            records.append({
                "date": dt.date(),
                "regime": result["regime"],
                "growth_score": round(result["growth_score"], 4),
                "inflation_score": round(result["inflation_score"], 4),
                "confidence": result["confidence"],
                "transition_warning": result["transition_warning"],
            })
        except Exception as e:
            records.append({
                "date": dt.date(), "regime": "ERROR",
                "growth_score": None, "inflation_score": None,
                "confidence": None, "transition_warning": None,
            })

    df = pd.DataFrame(records)
    df.to_csv(out_path, index=False)
    print(f"✓ Saved {out_path}  ({len(df)} monthly records)")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch FRED data and refresh parquet + regime history."
    )
    parser.add_argument("--start",           default="2000-01-01",
                        help="Start date for FRED fetch (default: 2000-01-01)")
    parser.add_argument("--out-dir",         default=str(REPO_ROOT / "data"),
                        help="Directory for fred_raw.parquet output")
    parser.add_argument("--refresh-history", action="store_true",
                        help="Also regenerate regime_history.csv")
    args = parser.parse_args()

    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        print("ERROR: FRED_API_KEY environment variable is not set.")
        print("  export FRED_API_KEY=your_key_here")
        sys.exit(1)

    out_path = Path(args.out_dir) / "fred_raw.parquet"
    loader = fetch_and_save(api_key, args.start, out_path)

    if args.refresh_history:
        history_path = REPO_ROOT / "regime_history.csv"
        refresh_regime_history(loader, args.start, history_path)

    print("\n✓ Done. Classifier and dashboard are now fully offline.")
    print("  Run  python dashboard.py  (no API key needed)")


if __name__ == "__main__":
    main()
