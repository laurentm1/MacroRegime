"""
loaders.py — Pluggable data loader pattern
===========================================
Three loader classes share the same interface (SeriesLoader ABC).

  ParquetLoader  (DEFAULT — offline)
      Reads from data/fred_raw.parquet written by scripts/update_data.py.
      No network, no API key required.  O(1) after the first load.

  FredLoader     (online fallback)
      Pulls live from FRED via fredapi.  Requires FRED_API_KEY env var.
      Used only by scripts/update_data.py and as an emergency fallback.

  MacroDataLoader  (legacy alias — kept for backward compatibility)
      Points at ParquetLoader by default; falls back to FredLoader if
      data/fred_raw.parquet is missing.

Usage
-----
    # offline (default, no API key needed):
    from loaders import ParquetLoader
    loader = ParquetLoader()

    # online refresh:
    from loaders import FredLoader
    loader = FredLoader(api_key=os.environ['FRED_API_KEY'])

    # Both expose identical methods:
    loader.get(name)                         -> pd.Series (full history)
    loader.as_of(name, date)                 -> float
    loader.yoy(name, date)                   -> float
    loader.mom_change(name, date, periods=1) -> float
    loader.rolling_percentile(name, date, window_years=5) -> float
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# ── Canonical series dict (single source of truth) ──────────────────────────
SERIES = {
    # PRIMARY — Growth
    "GDP":          "A191RL1Q225SBEA",
    "CFNAI":        "CFNAIMA3",
    "PAYROLLS":     "PAYEMS",
    "UNEMPLOYMENT": "UNRATE",
    # PRIMARY — Inflation
    "CORE_PCE":     "PCEPILFE",
    "CORE_CPI":     "CPILFESL",
    "CPI":          "CPIAUCSL",
    "BREAKEVEN":    "T10YIE",
    # LEADING — Transition signals
    "CREDIT":       "BAMLH0A0HYM2",
    "OIL":          "DCOILWTICO",
    "YIELD_2Y":     "DGS2",
    "YIELD_10Y":    "GS10",
    "VIX":          "VIXCLS",
    # CONFIRMATORY
    "FED_FUNDS":    "FEDFUNDS",
    "PPI":          "PPIACO",
    "WAGES":        "CES0500000003",
    "M2":           "M2SL",
    "MORTGAGE":     "MORTGAGE30US",
}

REPO_ROOT  = Path(__file__).parent
PARQUET_PATH = REPO_ROOT / "data" / "fred_raw.parquet"

# ── ABC ───────────────────────────────────────────────────────────────────────

class SeriesLoader(ABC):
    """Abstract base: every loader must implement get()."""

    @abstractmethod
    def get(self, name: str) -> pd.Series:
        ...

    # ── Shared derived methods (same for all loaders) ─────────────────────

    def load_all(self):
        print("Loading data...")
        for name in SERIES:
            self.get(name)
            print(f"  ✓ {name}")
        print("All series loaded.\n")

    def as_of(self, name: str, date) -> float:
        s = self.get(name)
        s_to = s[s.index <= pd.Timestamp(date)].dropna()
        return float(s_to.iloc[-1]) if len(s_to) else np.nan

    def yoy(self, name: str, date) -> float:
        s = self.get(name)
        s_to = s[s.index <= pd.Timestamp(date)].dropna()
        if len(s_to) < 13:
            return np.nan
        current = s_to.iloc[-1]
        year_ago_idx = s_to.index[-1] - pd.DateOffset(months=12)
        prior = s[s.index <= year_ago_idx].dropna()
        if len(prior) == 0:
            return np.nan
        return float((current / prior.iloc[-1] - 1) * 100)

    def mom_change(self, name: str, date, periods: int = 1) -> float:
        s = self.get(name)
        s_to = s[s.index <= pd.Timestamp(date)].dropna()
        if len(s_to) < periods + 1:
            return np.nan
        return float(s_to.iloc[-1] - s_to.iloc[-(periods + 1)])

    def rolling_percentile(self, name: str, date, window_years: int = 5) -> float:
        s = self.get(name)
        s_to = s[s.index <= pd.Timestamp(date)].dropna()
        if len(s_to) < 24:
            return np.nan
        window = window_years * 12
        recent = s_to.iloc[-window:] if len(s_to) >= window else s_to
        current = s_to.iloc[-1]
        return float((recent < current).sum() / len(recent))


# ── ParquetLoader (default / offline) ────────────────────────────────────────

class ParquetLoader(SeriesLoader):
    """
    Reads from data/fred_raw.parquet written by scripts/update_data.py.
    No network or API key required.  Loads the whole file once into memory
    then serves individual series from the in-memory cache.
    """

    def __init__(self, path: Path | str = PARQUET_PATH):
        self._path = Path(path)
        self._df: pd.DataFrame | None = None
        self._cache: dict[str, pd.Series] = {}

    def _load(self):
        if self._df is None:
            if not self._path.exists():
                raise FileNotFoundError(
                    f"Parquet file not found: {self._path}\n"
                    "Run  python scripts/update_data.py  to fetch data from FRED."
                )
            self._df = pd.read_parquet(self._path)
            self._df.index = pd.to_datetime(self._df.index)

    def get(self, name: str) -> pd.Series:
        if name not in self._cache:
            self._load()
            if name not in self._df.columns:
                raise KeyError(f"Series '{name}' not found in parquet. Re-run update_data.py.")
            self._cache[name] = self._df[name].dropna()
        return self._cache[name]


# ── FredLoader (online — used only by update_data.py) ────────────────────────

class FredLoader(SeriesLoader):
    """
    Live FRED fetcher.  Requires fredapi and FRED_API_KEY.
    Only import here keeps fredapi out of the main classifier/dashboard.
    """

    def __init__(self, api_key: str | None = None, start: str = "2000-01-01"):
        try:
            from fredapi import Fred  # lazy import — not required for offline use
        except ImportError:
            raise ImportError(
                "fredapi is not installed.  Install it with:\n"
                "  pip install fredapi\n"
                "It is only needed for data refresh (scripts/update_data.py)."
            )
        key = api_key or os.environ.get("FRED_API_KEY")
        if not key:
            raise ValueError(
                "FRED_API_KEY environment variable is not set.\n"
                "Export it before running update_data.py:\n"
                "  export FRED_API_KEY=your_key_here"
            )
        self._fred = Fred(api_key=key)
        self._start = start
        self._cache: dict[str, pd.Series] = {}

    def get(self, name: str) -> pd.Series:
        if name not in self._cache:
            code = SERIES[name]
            s = self._fred.get_series(code, observation_start=self._start)
            s.name = name
            self._cache[name] = s
        return self._cache[name]


# ── MacroDataLoader — legacy alias ────────────────────────────────────────────

def MacroDataLoader(api_key: str | None = None, start: str = "2000-01-01") -> SeriesLoader:
    """
    Drop-in replacement for the old MacroDataLoader class.
    Returns a ParquetLoader if data/fred_raw.parquet exists,
    otherwise falls back to FredLoader (prints a warning).
    Pass api_key only if you want to force online mode.
    """
    if api_key is None and PARQUET_PATH.exists():
        return ParquetLoader()
    if api_key is not None:
        print("[loaders] Using FredLoader (api_key supplied).")
        return FredLoader(api_key=api_key, start=start)
    # No parquet, no key → try env var
    env_key = os.environ.get("FRED_API_KEY")
    if env_key:
        print("[loaders] data/fred_raw.parquet not found — falling back to FRED live.")
        return FredLoader(api_key=env_key, start=start)
    raise FileNotFoundError(
        "data/fred_raw.parquet is missing and FRED_API_KEY is not set.\n"
        "Run  python scripts/update_data.py  first."
    )
