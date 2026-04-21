"""
Macro Regime Classifier
=======================
Classifies the US macro regime into one of four states:
  - GOLDILOCKS       : Growth above trend, inflation at/below target
  - REFLATION        : Growth accelerating, inflation rising
  - STAGFLATION_RISK : Growth decelerating/below trend, inflation sticky/rising
  - DEFLATION_RISK   : Growth contracting, inflation falling

Works fully offline — NO fredapi import, NO API key required.
Data is loaded via a SeriesLoader (ParquetLoader by default).
To refresh data run:  python scripts/update_data.py

Output per date:
  - regime label
  - growth_score    (-1 to +1)
  - inflation_score (-1 to +1)
  - confidence      (HIGH / MEDIUM / LOW)
  - transition_warning (bool)
  - signal breakdown dict
"""

import numpy as np
import pandas as pd
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Re-export SERIES and MacroDataLoader from loaders so existing
# code that does  `from regime_classifier import MacroDataLoader, SERIES`
# keeps working without changes.
from loaders import MacroDataLoader, SERIES  # noqa: F401  (public API)


# ── Regime Classifier ─────────────────────────────────────────────────────────

class RegimeClassifier:

    # ── Thresholds ────────────────────────────────────────────────────────────
    CORE_PCE_HIGH     = 2.5    # % YoY — above = inflation concern
    CORE_PCE_LOW      = 1.5    # % YoY — below = deflation concern
    CPI_HIGH          = 2.5
    CFNAI_EXPANSION   = 0.0    # CFNAI > 0 = above-trend growth
    CFNAI_CONTRACTION = -0.7   # CFNAI < -0.7 = recession territory
    CREDIT_WIDE       = 4.5    # HY spread > 4.5% = stress
    CREDIT_VERY_WIDE  = 7.0    # HY spread > 7.0% = crisis
    VIX_ELEVATED      = 25
    VIX_CRISIS        = 40
    OIL_SHOCK_PCT     = 20     # Oil up >20% in 3 months = supply shock flag

    def __init__(self, loader=None):
        """
        Parameters
        ----------
        loader : SeriesLoader, optional
            Any object implementing the SeriesLoader interface (ParquetLoader,
            FredLoader, or a custom mock).  Defaults to MacroDataLoader() which
            returns a ParquetLoader when fred_raw.parquet exists.
        """
        if loader is None:
            loader = MacroDataLoader()
        self.data = loader

    def classify(self, date) -> dict:
        date = pd.Timestamp(date)

        # ── Growth Score ──────────────────────────────────────────────────────
        cfnai        = self.data.as_of("CFNAI", date)
        gdp          = self.data.as_of("GDP", date)
        unemp        = self.data.as_of("UNEMPLOYMENT", date)
        unemp_3m_chg = self.data.mom_change("UNEMPLOYMENT", date, 3)
        payrolls_mom = self.data.mom_change("PAYROLLS", date, 1)

        cfnai_score = np.clip(cfnai / 1.5, -1, 1)            if not np.isnan(cfnai)        else 0.0
        gdp_score   = np.clip((gdp - 2.0) / 3.0, -1, 1)      if not np.isnan(gdp)          else 0.0
        unemp_score = np.clip(-unemp_3m_chg / 0.5, -1, 1)    if not np.isnan(unemp_3m_chg) else 0.0
        pay_score   = np.clip(payrolls_mom / 200, -1, 1)      if not np.isnan(payrolls_mom) else 0.0

        growth_score = (cfnai_score * 0.40 + gdp_score * 0.25 +
                        unemp_score * 0.20 + pay_score * 0.15)
        growth_score = float(np.clip(growth_score, -1, 1))

        # ── Inflation Score ───────────────────────────────────────────────────
        core_pce_yoy = self.data.yoy("CORE_PCE", date)
        core_cpi_yoy = self.data.yoy("CORE_CPI", date)
        breakeven    = self.data.as_of("BREAKEVEN", date)
        wages_yoy    = self.data.yoy("WAGES", date)
        ppi_yoy      = self.data.yoy("PPI", date)

        pce_score  = np.clip((core_pce_yoy - 2.5) / 2.0, -1, 1) if not np.isnan(core_pce_yoy) else 0.0
        cpi_score  = np.clip((core_cpi_yoy - 2.5) / 2.0, -1, 1) if not np.isnan(core_cpi_yoy) else 0.0
        be_score   = np.clip((breakeven - 2.5) / 1.0,    -1, 1)  if not np.isnan(breakeven)    else 0.0
        wage_score = np.clip((wages_yoy - 3.5) / 2.0,    -1, 1)  if not np.isnan(wages_yoy)    else 0.0

        inflation_score = (pce_score * 0.40 + cpi_score * 0.25 +
                           be_score  * 0.20 + wage_score * 0.15)
        inflation_score = float(np.clip(inflation_score, -1, 1))

        # ── Quadrant classification ───────────────────────────────────────────
        #   G+  I-  → GOLDILOCKS
        #   G+  I+  → REFLATION
        #   G-  I+  → STAGFLATION_RISK
        #   G-  I-  → DEFLATION_RISK
        if growth_score >= 0 and inflation_score < 0:
            regime = "GOLDILOCKS"
        elif growth_score >= 0 and inflation_score >= 0:
            regime = "REFLATION"
        elif growth_score < 0 and inflation_score >= 0:
            regime = "STAGFLATION_RISK"
        else:
            regime = "DEFLATION_RISK"

        # ── Leading / transition signals ──────────────────────────────────────
        credit_spread  = self.data.as_of("CREDIT", date)
        vix            = self.data.as_of("VIX", date)
        oil_now        = self.data.as_of("OIL", date)
        oil_3m_ago     = self.data.as_of("OIL", date - pd.DateOffset(months=3))
        oil_3m_chg_pct = ((oil_now - oil_3m_ago) / max(abs(oil_3m_ago), 1)) * 100 \
                         if not (np.isnan(oil_now) or np.isnan(oil_3m_ago)) else np.nan

        credit_stress    = bool(credit_spread > self.CREDIT_WIDE)    if not np.isnan(credit_spread) else False
        credit_crisis    = bool(credit_spread > self.CREDIT_VERY_WIDE) if not np.isnan(credit_spread) else False
        vix_elevated     = bool(vix > self.VIX_ELEVATED)             if not np.isnan(vix)           else False
        oil_shock        = bool(oil_3m_chg_pct > self.OIL_SHOCK_PCT) if not np.isnan(oil_3m_chg_pct) else False

        transition_warning = credit_stress or vix_elevated or oil_shock

        # Override: credit crisis + VIX spike → DEFLATION_RISK regardless
        if credit_crisis and not np.isnan(vix) and vix > self.VIX_CRISIS:
            regime = "DEFLATION_RISK"
            transition_warning = True

        # ── Confidence ────────────────────────────────────────────────────────
        avg_strength = (abs(growth_score) + abs(inflation_score)) / 2
        if avg_strength > 0.4:
            confidence = "HIGH"
        elif avg_strength > 0.2:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        # ── Yield curve ───────────────────────────────────────────────────────
        y2  = self.data.as_of("YIELD_2Y", date)
        y10 = self.data.as_of("YIELD_10Y", date)
        spread_2s10s = float(y10 - y2) if not (np.isnan(y10) or np.isnan(y2)) else np.nan
        inverted     = bool(spread_2s10s < 0) if not np.isnan(spread_2s10s) else False

        def _r(x, n=3):
            return round(float(x), n) if not np.isnan(x) else None

        return {
            "date":               date.strftime("%Y-%m-%d"),
            "regime":             regime,
            "growth_score":       round(growth_score, 3),
            "inflation_score":    round(inflation_score, 3),
            "confidence":         confidence,
            "transition_warning": transition_warning,
            "signals": {
                "cfnai":                _r(cfnai),
                "gdp_qoq":              _r(gdp, 2),
                "unemployment":         _r(unemp, 2),
                "unemp_3m_chg":         _r(unemp_3m_chg, 2),
                "core_pce_yoy":         _r(core_pce_yoy, 2),
                "core_cpi_yoy":         _r(core_cpi_yoy, 2),
                "breakeven_10y":        _r(breakeven, 2),
                "wages_yoy":            _r(wages_yoy, 2),
                "credit_spread":        _r(credit_spread, 2),
                "vix":                  _r(vix, 2),
                "oil_3m_chg_pct":       _r(oil_3m_chg_pct, 1),
                "spread_2s10s":         _r(spread_2s10s, 2),
                "yield_curve_inverted": inverted,
                "ppi_yoy":              _r(ppi_yoy, 2),
            },
        }


# ── Backtest helper ───────────────────────────────────────────────────────────

def run_backtest(start: str = "2000-01-01", end: str | None = None,
                 freq: str = "MS", loader=None) -> pd.DataFrame:
    """
    Run the classifier on every period from start → end.
    Returns a DataFrame indexed by date with regime labels and scores.
    Uses ParquetLoader by default (offline); pass a FredLoader to use live data.
    """
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    if loader is None:
        loader = MacroDataLoader()
    loader.load_all()
    classifier = RegimeClassifier(loader=loader)

    dates = pd.date_range(start=start, end=end, freq=freq)
    results = []
    print(f"Running classifier on {len(dates)} dates...")
    for i, dt in enumerate(dates):
        try:
            r = classifier.classify(dt)
            results.append(r)
            if i % 12 == 0:
                print(f"  {r['date']}  →  {r['regime']:20}  "
                      f"(G:{r['growth_score']:+.2f} I:{r['inflation_score']:+.2f} "
                      f"conf:{r['confidence']})")
        except Exception as e:
            print(f"  ✗ {dt.strftime('%Y-%m-%d')} — {e}")

    df = pd.DataFrame(results)
    signals_df = pd.json_normalize(df["signals"])
    df = pd.concat([df.drop("signals", axis=1), signals_df], axis=1)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df


if __name__ == "__main__":
    df = run_backtest(start="2000-01-01")
    out = "regime_history.csv"
    df.to_csv(out)
    print(f"\nSaved {out} — {len(df)} rows")
    print("\nRegime distribution:")
    print(df["regime"].value_counts())
    print("\nLast 12 months:")
    print(df[["regime", "growth_score", "inflation_score",
              "confidence", "transition_warning"]].tail(12).to_string())
