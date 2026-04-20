"""
Macro Regime Classifier
=======================
Classifies the US macro regime into one of four states:
  - GOLDILOCKS       : Growth above trend, inflation at/below target
  - REFLATION        : Growth accelerating, inflation rising
  - STAGFLATION_RISK : Growth decelerating/below trend, inflation sticky/rising
  - DEFLATION_RISK   : Growth contracting, inflation falling

Uses 19 FRED series. Works on any historical date back to 2000.

Output per date:
  - regime label
  - growth_score   (-1 to +1)
  - inflation_score (-1 to +1)
  - confidence      (HIGH / MEDIUM / LOW)
  - transition_warning (bool)
  - signal breakdown
"""

import pandas as pd
import numpy as np
from fredapi import Fred
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

FRED_API_KEY = 'b1116b7ae7351cdf8018805fbc612ec3'

# ── FRED Series ──────────────────────────────────────────────────────────────

SERIES = {
    # PRIMARY — Growth
    "GDP":          "A191RL1Q225SBEA",   # Real GDP QoQ annualised (quarterly)
    "CFNAI":        "CFNAIMA3",          # Chicago Fed NAI 3M MA (composite of 85 indicators)
    "PAYROLLS":     "PAYEMS",            # Nonfarm payrolls level
    "UNEMPLOYMENT": "UNRATE",            # Unemployment rate

    # PRIMARY — Inflation
    "CORE_PCE":     "PCEPILFE",          # Core PCE YoY (Fed's preferred)
    "CORE_CPI":     "CPILFESL",          # Core CPI YoY
    "CPI":          "CPIAUCSL",          # CPI headline
    "BREAKEVEN":    "T10YIE",            # 10Y inflation breakeven

    # LEADING — Transition signals
    "CREDIT":       "BAMLH0A0HYM2",      # HY credit spreads (leads GDP by 4-8 weeks)
    "OIL":          "DCOILWTICO",        # WTI crude (supply shock detector)
    "YIELD_2Y":     "DGS2",             # 2Y Treasury yield (daily)
    "YIELD_10Y":    "GS10",             # 10Y Treasury yield
    "VIX":          "VIXCLS",           # Equity volatility

    # CONFIRMATORY
    "FED_FUNDS":    "FEDFUNDS",          # Fed Funds rate
    "PPI":          "PPIACO",            # PPI YoY
    "WAGES":        "CES0500000003",     # Avg hourly earnings
    "M2":           "M2SL",             # M2 money supply
    "MORTGAGE":     "MORTGAGE30US",      # 30Y mortgage rate
}

# ── Data Loader ───────────────────────────────────────────────────────────────

class MacroDataLoader:
    def __init__(self, api_key=FRED_API_KEY, start='2000-01-01'):
        self.fred = Fred(api_key=api_key)
        self.start = start
        self._cache = {}

    def get(self, name):
        if name not in self._cache:
            code = SERIES[name]
            s = self.fred.get_series(code, observation_start=self.start)
            self._cache[name] = s
        return self._cache[name]

    def load_all(self):
        print("Loading FRED data...")
        for name in SERIES:
            self.get(name)
            print(f"  ✓ {name}")
        print("All series loaded.\n")

    def as_of(self, name, date):
        """Return the most recent available value as of a given date (no look-ahead)."""
        s = self.get(name)
        s_to_date = s[s.index <= pd.Timestamp(date)].dropna()
        if len(s_to_date) == 0:
            return np.nan
        return s_to_date.iloc[-1]

    def yoy(self, name, date):
        """Year-over-year % change as of date."""
        s = self.get(name)
        s_to_date = s[s.index <= pd.Timestamp(date)].dropna()
        if len(s_to_date) < 13:
            return np.nan
        current = s_to_date.iloc[-1]
        year_ago_idx = s_to_date.index[-1] - pd.DateOffset(months=12)
        prior = s[s.index <= year_ago_idx].dropna()
        if len(prior) == 0:
            return np.nan
        return (current / prior.iloc[-1] - 1) * 100

    def mom_change(self, name, date, periods=1):
        """Change over N periods as of date."""
        s = self.get(name)
        s_to_date = s[s.index <= pd.Timestamp(date)].dropna()
        if len(s_to_date) < periods + 1:
            return np.nan
        return s_to_date.iloc[-1] - s_to_date.iloc[-(periods+1)]

    def rolling_percentile(self, name, date, window_years=5):
        """Where does the current value rank in the trailing N-year distribution."""
        s = self.get(name)
        s_to_date = s[s.index <= pd.Timestamp(date)].dropna()
        if len(s_to_date) < 24:
            return np.nan
        window = window_years * 12
        recent_window = s_to_date.iloc[-window:] if len(s_to_date) >= window else s_to_date
        current = s_to_date.iloc[-1]
        pct = (recent_window < current).mean()
        return pct  # 0 = historically low, 1 = historically high


# ── Regime Classifier ────────────────────────────────────────────────────────

class RegimeClassifier:

    # Thresholds
    CORE_PCE_HIGH    = 2.5   # % YoY — above = inflation concern
    CORE_PCE_LOW     = 1.5   # % YoY — below = deflation concern
    CPI_HIGH         = 2.5
    CFNAI_EXPANSION  = 0.0   # CFNAI > 0 = above-trend growth
    CFNAI_CONTRACTION = -0.7 # CFNAI < -0.7 = recession territory
    CREDIT_WIDE      = 4.5   # HY spread > 4.5% = stress
    CREDIT_VERY_WIDE = 7.0   # HY spread > 7.0% = crisis
    VIX_ELEVATED     = 25
    VIX_CRISIS       = 40
    OIL_SHOCK_PCT    = 20    # Oil up >20% in 3 months = supply shock flag

    def __init__(self, loader: MacroDataLoader):
        self.data = loader

    def classify(self, date) -> dict:
        date = pd.Timestamp(date)

        # ── Growth Score ────────────────────────────────────────────────────
        cfnai        = self.data.as_of('CFNAI', date)
        gdp          = self.data.as_of('GDP', date)
        unemp        = self.data.as_of('UNEMPLOYMENT', date)
        unemp_3m_chg = self.data.mom_change('UNEMPLOYMENT', date, 3)
        payrolls_mom = self.data.mom_change('PAYROLLS', date, 1)

        # CFNAI: centred at 0 (above trend = positive)
        cfnai_score = np.clip(cfnai / 1.5, -1, 1) if not np.isnan(cfnai) else 0

        # GDP: annualised, trend ~2%
        gdp_score = np.clip((gdp - 2.0) / 3.0, -1, 1) if not np.isnan(gdp) else 0

        # Unemployment direction: rising = bad
        unemp_score = np.clip(-unemp_3m_chg / 0.5, -1, 1) if not np.isnan(unemp_3m_chg) else 0

        # Payrolls: positive momentum = good
        pay_score = np.clip(payrolls_mom / 200, -1, 1) if not np.isnan(payrolls_mom) else 0

        growth_score = (cfnai_score * 0.40 + gdp_score * 0.25 +
                        unemp_score * 0.20 + pay_score * 0.15)
        growth_score = np.clip(growth_score, -1, 1)

        # ── Inflation Score ──────────────────────────────────────────────────
        core_pce_yoy  = self.data.yoy('CORE_PCE', date)
        core_cpi_yoy  = self.data.yoy('CORE_CPI', date)
        cpi_yoy       = self.data.yoy('CPI', date)
        breakeven     = self.data.as_of('BREAKEVEN', date)
        wages_yoy     = self.data.yoy('WAGES', date)
        ppi_yoy       = self.data.yoy('PPI', date)

        # Score relative to 2.5% target
        pce_score  = np.clip((core_pce_yoy - 2.5) / 2.0, -1, 1)  if not np.isnan(core_pce_yoy)  else 0
        cpi_score  = np.clip((core_cpi_yoy - 2.5) / 2.0, -1, 1)  if not np.isnan(core_cpi_yoy)  else 0
        be_score   = np.clip((breakeven - 2.5) / 1.0, -1, 1)      if not np.isnan(breakeven)      else 0
        wage_score = np.clip((wages_yoy - 3.5) / 2.0, -1, 1)      if not np.isnan(wages_yoy)      else 0

        inflation_score = (pce_score * 0.40 + cpi_score * 0.25 +
                           be_score * 0.20 + wage_score * 0.15)
        inflation_score = np.clip(inflation_score, -1, 1)

        # ── Regime Classification ────────────────────────────────────────────
        #
        #   inflation_score > 0  = inflation ABOVE target (sticky/rising)
        #   inflation_score < 0  = inflation BELOW target (falling/low)
        #   growth_score > 0     = growth ABOVE trend
        #   growth_score < 0     = growth BELOW trend
        #
        #   Quadrant:
        #     G+, I-  → GOLDILOCKS
        #     G+, I+  → REFLATION
        #     G-, I+  → STAGFLATION_RISK
        #     G-, I-  → DEFLATION_RISK

        if growth_score >= 0 and inflation_score < 0:
            regime = "GOLDILOCKS"
        elif growth_score >= 0 and inflation_score >= 0:
            regime = "REFLATION"
        elif growth_score < 0 and inflation_score >= 0:
            regime = "STAGFLATION_RISK"
        else:
            regime = "DEFLATION_RISK"

        # ── Leading Signals (transition warnings) ────────────────────────────
        credit_spread    = self.data.as_of('CREDIT', date)
        vix              = self.data.as_of('VIX', date)
        oil_3m_chg_pct   = (self.data.mom_change('OIL', date, 3) /
                            max(abs(self.data.as_of('OIL', date) - self.data.mom_change('OIL', date, 3)), 1)) * 100

        credit_stress    = credit_spread > self.CREDIT_WIDE    if not np.isnan(credit_spread) else False
        credit_crisis    = credit_spread > self.CREDIT_VERY_WIDE if not np.isnan(credit_spread) else False
        vix_elevated     = vix > self.VIX_ELEVATED             if not np.isnan(vix) else False
        oil_shock        = oil_3m_chg_pct > self.OIL_SHOCK_PCT if not np.isnan(oil_3m_chg_pct) else False

        transition_warning = credit_stress or vix_elevated or oil_shock

        # Override to DEFLATION_RISK in a credit crisis regardless of other signals
        if credit_crisis and vix > self.VIX_CRISIS:
            regime = "DEFLATION_RISK"
            transition_warning = True

        # ── Confidence Score ──────────────────────────────────────────────────
        # Confidence is high when growth and inflation scores are unambiguous
        # (both clearly positive or negative), low when near zero
        g_strength = abs(growth_score)
        i_strength = abs(inflation_score)
        avg_strength = (g_strength + i_strength) / 2

        if avg_strength > 0.4:
            confidence = "HIGH"
        elif avg_strength > 0.2:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        # ── Yield Curve ───────────────────────────────────────────────────────
        y2  = self.data.as_of('YIELD_2Y', date)
        y10 = self.data.as_of('YIELD_10Y', date)
        spread_2s10s = (y10 - y2) if not (np.isnan(y10) or np.isnan(y2)) else np.nan
        inverted = spread_2s10s < 0 if not np.isnan(spread_2s10s) else False

        return {
            "date":               date.strftime('%Y-%m-%d'),
            "regime":             regime,
            "growth_score":       round(growth_score, 3),
            "inflation_score":    round(inflation_score, 3),
            "confidence":         confidence,
            "transition_warning": transition_warning,
            # Raw signals
            "signals": {
                "cfnai":           round(cfnai, 3)          if not np.isnan(cfnai) else None,
                "gdp_qoq":         round(gdp, 2)            if not np.isnan(gdp) else None,
                "unemployment":    round(unemp, 2)          if not np.isnan(unemp) else None,
                "unemp_3m_chg":    round(unemp_3m_chg, 2)  if not np.isnan(unemp_3m_chg) else None,
                "core_pce_yoy":    round(core_pce_yoy, 2)  if not np.isnan(core_pce_yoy) else None,
                "core_cpi_yoy":    round(core_cpi_yoy, 2)  if not np.isnan(core_cpi_yoy) else None,
                "breakeven_10y":   round(breakeven, 2)      if not np.isnan(breakeven) else None,
                "wages_yoy":       round(wages_yoy, 2)      if not np.isnan(wages_yoy) else None,
                "credit_spread":   round(credit_spread, 2)  if not np.isnan(credit_spread) else None,
                "vix":             round(vix, 2)            if not np.isnan(vix) else None,
                "oil_3m_chg_pct":  round(oil_3m_chg_pct, 1) if not np.isnan(oil_3m_chg_pct) else None,
                "spread_2s10s":    round(spread_2s10s, 2)  if not np.isnan(spread_2s10s) else None,
                "yield_curve_inverted": inverted,
            }
        }


# ── Backtest Engine ───────────────────────────────────────────────────────────

def run_backtest(start='2000-01-01', end=None, freq='MS'):
    """
    Run the classifier on every month-start from start to end.
    Returns a DataFrame with regime labels and scores.
    """
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')

    loader = MacroDataLoader(start=start)
    loader.load_all()
    classifier = RegimeClassifier(loader)

    dates = pd.date_range(start=start, end=end, freq=freq)
    results = []

    print(f"Running classifier on {len(dates)} dates...")
    for i, date in enumerate(dates):
        try:
            result = classifier.classify(date)
            results.append(result)
            if i % 12 == 0:
                print(f"  {result['date']}  →  {result['regime']:20}  "
                      f"(G:{result['growth_score']:+.2f} I:{result['inflation_score']:+.2f} "
                      f"conf:{result['confidence']})")
        except Exception as e:
            print(f"  ✗ {date.strftime('%Y-%m-%d')} — {e}")

    df = pd.DataFrame(results)
    # Flatten signals column
    signals_df = pd.json_normalize(df['signals'])
    df = pd.concat([df.drop('signals', axis=1), signals_df], axis=1)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df


if __name__ == '__main__':
    df = run_backtest(start='2000-01-01')
    df.to_csv('/home/user/workspace/macro_regime/regime_history.csv')
    print(f"\nSaved to regime_history.csv — {len(df)} rows")
    print("\nRegime distribution:")
    print(df['regime'].value_counts())
    print("\nLast 12 months:")
    print(df[['regime','growth_score','inflation_score','confidence','transition_warning']].tail(12).to_string())
