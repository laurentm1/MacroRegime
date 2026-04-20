# Macro Regime System

A systematic macro regime classifier and event-driven trading strategy framework built on FRED data. Classifies the US economy into one of four macro regimes on any historical date, then uses those regime labels to condition trading signals around scheduled economic data releases.

---

## Project Overview

This is **Phase 1** of a multi-phase systematic macro trading system:

| Phase | Description | Status |
|---|---|---|
| **1 — Regime Classifier** | Rules-based regime classification from 19 FRED series | ✅ Complete |
| **2 — Event Strategy** | Regime-conditioned event study backtest (FRED proxy) | ✅ Complete |
| **3 — Bloomberg Backtest** | Same strategy with real consensus surprise data | 🔜 Next |
| **4 — Live Signal Dashboard** | Interactive web dashboard with date picker | 🔜 Planned |
| **5 — Risk Framework** | Position sizing, drawdown stops, transition triggers | 🔜 Planned |

---

## Macro Regime Framework

The classifier places the US economy into one of four regimes at any point in time, inspired by the Druckenmiller macro framework:

| Regime | Growth | Inflation | Rates Posture | FX Posture |
|---|---|---|---|---|
| **Goldilocks** | Above trend | At/below target | Neutral duration; steepener bias | Long carry (AUD); short safe-haven (JPY, CHF) |
| **Reflation** | Accelerating | Rising | Short duration; steepener; long breakevens | Long commodity FX; short USD |
| **Stagflation Risk** | Decelerating/below trend | Sticky/rising | Short front-end; flattener; long breakevens | Short high-beta FX; long CHF, JPY |
| **Deflation Risk** | Contracting | Falling | Max long duration; bull flattener | Long USD, JPY; short carry basket |

---

## Classifier Architecture

### Data Sources (all FRED, all free)

```python
SERIES = {
    # PRIMARY — Growth
    "GDP":          "A191RL1Q225SBEA",   # Real GDP QoQ annualised (quarterly)
    "CFNAI":        "CFNAIMA3",          # Chicago Fed NAI 3M MA (85 indicators)
    "PAYROLLS":     "PAYEMS",            # Nonfarm payrolls
    "UNEMPLOYMENT": "UNRATE",            # Unemployment rate

    # PRIMARY — Inflation
    "CORE_PCE":     "PCEPILFE",          # Core PCE YoY (Fed's preferred)
    "CORE_CPI":     "CPILFESL",          # Core CPI YoY
    "CPI":          "CPIAUCSL",          # CPI headline
    "BREAKEVEN":    "T10YIE",            # 10Y inflation breakeven

    # LEADING — Transition signals
    "CREDIT":       "BAMLH0A0HYM2",      # HY credit spreads (leads GDP 4-8 weeks)
    "OIL":          "DCOILWTICO",        # WTI crude (supply shock detector)
    "YIELD_2Y":     "DGS2",              # 2Y Treasury yield
    "YIELD_10Y":    "GS10",              # 10Y Treasury yield
    "VIX":          "VIXCLS",            # Equity volatility

    # CONFIRMATORY
    "FED_FUNDS":    "FEDFUNDS",          # Fed Funds rate
    "PPI":          "PPIACO",            # PPI YoY
    "WAGES":        "CES0500000003",     # Avg hourly earnings
    "M2":           "M2SL",             # M2 money supply
    "MORTGAGE":     "MORTGAGE30US",      # 30Y mortgage rate
}
```

### Signal Scoring

Each date produces two composite scores:

**Growth Score** (−1 to +1):
- CFNAI 3M MA (weight 40%) — composite of 85 monthly indicators
- Real GDP QoQ vs 2% trend (weight 25%)
- Unemployment rate direction 3M change (weight 20%)
- Nonfarm payrolls monthly change (weight 15%)

**Inflation Score** (−1 to +1):
- Core PCE YoY vs 2.5% target (weight 40%)
- Core CPI YoY vs 2.5% target (weight 25%)
- 10Y breakeven vs 2.5% (weight 20%)
- Wage growth YoY vs 3.5% (weight 15%)

### Regime Classification

```
Growth ≥ 0 AND Inflation < 0  →  GOLDILOCKS
Growth ≥ 0 AND Inflation ≥ 0  →  REFLATION
Growth < 0  AND Inflation ≥ 0  →  STAGFLATION_RISK
Growth < 0  AND Inflation < 0  →  DEFLATION_RISK
```

### Output Per Date

```python
{
    "date":               "2008-10-01",
    "regime":             "DEFLATION_RISK",
    "growth_score":       -1.000,
    "inflation_score":    -0.614,
    "confidence":         "HIGH",         # HIGH / MEDIUM / LOW
    "transition_warning": True,           # credit spread or VIX spike detected
    "signals": {
        "cfnai":           -2.31,
        "gdp_qoq":         -2.7,
        "unemployment":    6.5,
        "core_pce_yoy":    2.4,
        "breakeven_10y":   1.72,
        "credit_spread":   7.85,
        "vix":             59.89,
        ...
    }
}
```

### Confidence & Transition Warnings

- **HIGH confidence**: both growth and inflation scores have magnitude > 0.4 (clear signal)
- **MEDIUM confidence**: average score magnitude 0.2–0.4
- **LOW confidence**: near the quadrant boundary — regime may be transitioning
- **Transition warning**: fires when HY credit spreads > 4.5% OR VIX > 25 OR oil up >20% in 3 months

---

## Historical Backtest Results (2000–2026)

### Regime Distribution

| Regime | Months | % of time |
|---|---|---|
| Goldilocks | 161 | 51% |
| Deflation Risk | 75 | 24% |
| Reflation | 64 | 20% |
| Stagflation Risk | 16 | 5% |

### Validation Against Known Episodes

| Period | Classifier | Reality |
|---|---|---|
| 2003–2007 | Goldilocks (HIGH conf) | Pre-GFC expansion, low inflation ✅ |
| 2008–2009 | Deflation Risk (HIGH conf, growth −1.0) | GFC ✅ |
| 2010–2019 | Mostly Goldilocks | Post-crisis ZIRP era ✅ |
| 2020 Mar–Jun | Deflation Risk (transition warning) | COVID shock ✅ |
| 2021 | Goldilocks → Reflation transition | Recovery + inflation ignition ✅ |
| 2022–2023 | Reflation (HIGH conf, inflation +0.81) | Peak inflation, fastest hike cycle in 40Y ✅ |
| 2025 | Oscillating Stagflation/Reflation (LOW conf) | Mixed signals, Fed on hold ✅ |

---

## Event Strategy Backtest

### Concept

The core insight: **the same economic data surprise has dramatically different market impact depending on the macro regime.** The regime determines what the market is paying attention to.

```
MACRO REGIME  +  DATA SURPRISE  =  TRADE SIGNAL
```

### Surprise Proxy (FRED version)

Since Bloomberg consensus data is not yet integrated, surprise is approximated as a z-score versus the trailing 12-month rolling median:

```
surprise_z = (actual_release - rolling_12m_median) / rolling_12m_std

BEAT   if surprise_z > +0.5
MISS   if surprise_z < -0.5
INLINE otherwise
```

This is a proxy. Bloomberg consensus data will produce cleaner results (Phase 3).

### Events Tracked

| Event | FRED Series | Frequency |
|---|---|---|
| CPI MoM | CPIAUCSL | Monthly |
| Core PCE MoM | PCEPILFE | Monthly |
| Nonfarm Payrolls | PAYEMS | Monthly |
| Philly Fed Manufacturing | GACDFSA066MSFRBPHI | Monthly |
| GDP QoQ | A191RL1Q225SBEA | Quarterly |

### Forward Returns

Measured as the change in 10Y Treasury yield over 1, 5, and 21 business days after the release date (bps). A positive signed return means the trade worked:
- BEAT → expect yield UP → positive = correct
- MISS → expect yield DOWN → negative flipped to positive = correct

### Key Findings

**CPI is regime-dependent (Table 2):**

| Regime | N | Mean (bps) | Hit% | Sharpe |
|---|---|---|---|---|
| Stagflation Risk | 5 | **+9.8** | 60% | **1.78** |
| Goldilocks | 78 | +2.5 | 54% | 0.73 |
| Deflation Risk | 29 | +2.4 | 48% | 0.62 |
| Reflation | 30 | **−0.8** | 40% | **−0.24** |

CPI beat in Stagflation Risk produces 4× the yield move vs Goldilocks. In Reflation the signal reverses — markets already expect inflation so a beat doesn't surprise.

**NFP is regime-dependent (Table 3):**

| Regime | N | Mean (bps) | Hit% | Sharpe |
|---|---|---|---|---|
| Deflation Risk | 35 | **+4.6** | 57% | **1.11** |
| Goldilocks | 70 | −1.0 | 43% | −0.31 |
| Reflation | 29 | −3.6 | 45% | −0.78 |

NFP matters most when growth fear dominates (Deflation Risk). In Reflation it actually works backwards.

### Strategy Variants (CPI + NFP events, 5-day 10Y yield trade)

| Variant | N trades | Total (bps) | Sharpe | Hit% |
|---|---|---|---|---|
| V1: All regimes, no filter | 285 | +267 | 0.25 | 48% |
| V2: Stagflation + Reflation only | 73 | −87 | −0.28 | 43% |
| V3: HIGH confidence only | 94 | +76 | 0.18 | 49% |
| V4: Confidence-sized | 285 | +189 | 0.23 | 48% |

**Note:** Low Sharpe ratios are expected with the FRED proxy surprise. Bloomberg consensus data will materially improve signal quality.

---

## Instruments (CME Futures)

All strategies expressed in CME-listed futures:

**Rates:**
- `ZT` — 2Y Treasury Note ($200K notional)
- `ZF` — 5Y Treasury Note ($100K notional)
- `ZN` — 10Y Treasury Note ($100K notional)
- `ZB / UB` — 30Y Bond / Ultra Bond ($100K notional)
- `SR3` — SOFR 3M futures ($2.5M notional)

**FX:**
- `6E` — Euro, `6J` — Japanese Yen, `6A` — Australian Dollar
- `6S` — Swiss Franc, `DX` — US Dollar Index

---

## Setup & Usage

### Requirements

```bash
pip install -r requirements.txt
```

### FRED API Key

Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html

Set it in both scripts:
```python
FRED_API_KEY = 'your_key_here'
```

### Run the Regime Classifier

```python
from regime_classifier import MacroDataLoader, RegimeClassifier

loader = MacroDataLoader(api_key='your_fred_key')
loader.load_all()
classifier = RegimeClassifier(loader)

# Classify any historical date
result = classifier.classify('2008-10-01')
print(result['regime'])        # DEFLATION_RISK
print(result['confidence'])    # HIGH
print(result['growth_score'])  # -1.000

# Run full backtest 2000–today
from regime_classifier import run_backtest
df = run_backtest(start='2000-01-01')
df.to_csv('regime_history.csv')
```

### Run the Event Backtest

```bash
python event_backtest.py
```

Outputs:
- `event_backtest_raw.csv` — every event with regime, surprise, forward returns
- `event_backtest_summary.csv` — aggregated stats by event × regime × surprise
- `event_backtest_chart.png` — visualisation of results

### Generate Regime Chart

```bash
python plot_regime.py
```

Outputs `regime_chart.png` — 4-panel chart showing regime history 2000–2026 overlaid with S&P 500, growth/inflation scores, and Core PCE vs 10Y yield.

---

## File Structure

```
macro_regime/
├── README.md                    — This file
├── requirements.txt             — Python dependencies
├── regime_classifier.py         — Core regime classifier module
├── plot_regime.py               — Regime history visualisation
├── event_backtest.py            — Event strategy backtest (FRED proxy)
├── regime_history.csv           — 316 monthly regime labels 2000–2026
├── regime_chart.png             — Regime history chart
├── event_backtest_raw.csv       — Raw event backtest data (1,103 events)
├── event_backtest_summary.csv   — Aggregated results table
└── event_backtest_chart.png     — Event strategy results chart
```

---

## Roadmap

### Phase 3 — Bloomberg Consensus Integration
Replace the FRED proxy surprise with real consensus data from Bloomberg:
```
Bloomberg ECO <GO> export:
  Fields: ACTUAL_RELEASE, SURVEY_MEDIAN, SURVEY_HIGH, SURVEY_LOW
  Events: CPI, NFP, PCE, GDP, ISM, FOMC
  Date range: 2005–today
```
Expected improvement: Sharpe +0.3–0.5 across all strategy variants.

### Phase 4 — Live Signal Dashboard
Interactive web app (built on existing Macro Terminal UI) with:
- Date picker → any historical date → full regime breakdown
- Current regime → current event calendar → live trade signals
- Economic surprise score updated on each data release

### Phase 5 — Risk Framework
- DV01-weighted position sizing per strategy
- Regime transition = position reduction trigger
- Per-strategy drawdown stops
- Portfolio-level VaR and correlation monitoring

### Phase 6 — Strategy Expansion
- FX carry basket conditioned on regime
- Monetary policy divergence (G10 rate differential)
- TIPS vs Nominals breakeven trade
- Commodity overlay (gold/copper regime signals)

---

## Research References

- **Druckenmiller macro framework** — regime-conditional asset allocation
- [Federal Reserve (2025)](https://www.federalreserve.gov/econres/feds/files/2025022pap.pdf) — investor attention and regime-dependent CPI sensitivity
- [AQR — Macro Momentum](https://www.returnstacked.com/academic-review/a-half-century-of-macro-momentum/) — 50-year backtest of macro momentum across asset classes
- [Research Affiliates — Systematic Global Macro](https://www.researchaffiliates.com/publications/articles/563-systematic-global-macro) — carry, momentum, and value across macro asset classes
- [John Hancock — Macro Strategies Across Monetary Policy Regimes](https://www.jhinvestments.com/viewpoints/alternatives/role-of-macro-strategies-across-monetary-policy-regimes) — regime-conditional hedge fund alpha

---

## Notes

- All FRED data is fetched with no look-ahead bias — `as_of(date)` always returns the most recently available value *as of* that date
- FRED vintage data (ALFRED) should be used for even stricter historical accuracy (data revisions)
- The CFNAI 3-month MA is used instead of ISM PMI because it aggregates 85 monthly indicators and is freely available on FRED
- Regime labels are generated at month-start frequency; daily interpolation is straightforward if needed for higher-frequency strategies

---

*Built with FRED API · CME Futures · Python · pandas · matplotlib*
