"""
Event Strategy Backtest — FRED Proxy Surprise
==============================================
Since we don't have Bloomberg consensus data yet, we approximate the
"surprise" as the deviation of each release from its recent trend
(12-month rolling median). This is a reasonable proxy: the market's
expectation of any release is roughly where the series has been trending.

Logic:
  surprise_proxy = actual_release - rolling_12m_median(series)
  surprise_direction = BEAT if surprise_proxy > +0.5 sigma, MISS if < -0.5 sigma, INLINE otherwise

Events tracked:
  1. CPI MoM
  2. Core PCE MoM
  3. NFP MoM change (000s)
  4. ISM Manufacturing
  5. GDP QoQ annualised

Forward returns measured in:
  - 10Y Treasury yield change (proxy for ZN P&L — yield UP = short ZN profitable)
  - 2s10s spread change (proxy for curve trade)
  - DGS2 yield change (proxy for ZT / front-end)
  Horizons: 1 day, 5 days, 21 days
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from fredapi import Fred
import warnings
warnings.filterwarnings('ignore')

FRED_API_KEY = 'b1116b7ae7351cdf8018805fbc612ec3'
REGIME_CSV   = '/home/user/workspace/macro_regime/regime_history.csv'
OUTPUT_DIR   = '/home/user/workspace/macro_regime/'

fred = Fred(api_key=FRED_API_KEY)

# ── Load regime history ───────────────────────────────────────────────────────
print("Loading regime history...")
regimes = pd.read_csv(REGIME_CSV, index_col='date', parse_dates=True)

def get_regime_as_of(date):
    """Return regime label for the most recent month-start on or before date."""
    available = regimes[regimes.index <= date]
    if len(available) == 0:
        return None
    return available.iloc[-1]['regime']

def get_confidence_as_of(date):
    available = regimes[regimes.index <= date]
    if len(available) == 0:
        return None
    return available.iloc[-1]['confidence']

# ── Load market data (daily) ──────────────────────────────────────────────────
print("Loading market data...")
y10  = fred.get_series('DGS10',  observation_start='2004-01-01').dropna()
y2   = fred.get_series('DGS2',   observation_start='2004-01-01').dropna()
sp500 = fred.get_series('SP500', observation_start='2004-01-01').dropna()

spread_2s10s = (y10 - y2).dropna()

def get_fwd_return(series, date, days):
    """
    Return the change in a yield series N business days after date.
    For yields: positive = yield rose = short bond profitable.
    """
    future = series[series.index > date]
    if len(future) < days:
        return np.nan
    return future.iloc[days-1] - series.asof(date)

# ── Load economic release series (monthly) ────────────────────────────────────
print("Loading economic series...")

# CPI MoM — compute from level series
cpi_raw      = fred.get_series('CPIAUCSL', observation_start='2004-01-01').dropna()
cpi_mom      = cpi_raw.pct_change(1) * 100   # MoM %

# Core PCE MoM
pce_raw      = fred.get_series('PCEPILFE', observation_start='2004-01-01').dropna()
pce_mom      = pce_raw.pct_change(1) * 100

# NFP — monthly change in thousands
nfp_raw      = fred.get_series('PAYEMS', observation_start='2004-01-01').dropna()
nfp_chg      = nfp_raw.diff(1)   # change in 000s

# ISM Manufacturing (use CFNAI as proxy — ISM not on FRED)
# We'll use the Philly Fed General Activity as a monthly manufacturing proxy
philly       = fred.get_series('GACDFSA066MSFRBPHI', observation_start='2004-01-01').dropna()

# GDP QoQ annualised
gdp_raw      = fred.get_series('A191RL1Q225SBEA', observation_start='2004-01-01').dropna()

EVENTS = {
    'CPI_MOM':   cpi_mom,
    'PCE_MOM':   pce_mom,
    'NFP':       nfp_chg,
    'PHILLY_FED':philly,
    'GDP_QOQ':   gdp_raw,
}

# ── Surprise Proxy ────────────────────────────────────────────────────────────

def compute_surprise_proxy(series, window=12):
    """
    Surprise proxy = (actual - rolling median) / rolling std
    Normalised z-score relative to trailing window.
    """
    rolling_med = series.rolling(window, min_periods=6).median()
    rolling_std = series.rolling(window, min_periods=6).std()
    z = (series - rolling_med) / rolling_std.replace(0, np.nan)
    return z

def classify_surprise(z):
    if pd.isna(z):
        return 'UNKNOWN'
    if z > 0.5:
        return 'BEAT'
    elif z < -0.5:
        return 'MISS'
    else:
        return 'INLINE'

# ── Build event table ─────────────────────────────────────────────────────────
print("Building event table...")

records = []
for event_name, series in EVENTS.items():
    surprise_z = compute_surprise_proxy(series)

    for date, val in series.items():
        if pd.isna(val):
            continue
        if date < pd.Timestamp('2005-01-01'):
            continue

        z     = surprise_z.asof(date) if date in surprise_z.index else np.nan
        surp  = classify_surprise(z)
        regime = get_regime_as_of(date)
        conf   = get_confidence_as_of(date)

        if regime is None:
            continue

        # Forward returns
        fwd_y10_1d  = get_fwd_return(y10,          date, 1)
        fwd_y10_5d  = get_fwd_return(y10,          date, 5)
        fwd_y10_21d = get_fwd_return(y10,          date, 21)
        fwd_2s10_1d = get_fwd_return(spread_2s10s, date, 1)
        fwd_2s10_5d = get_fwd_return(spread_2s10s, date, 5)
        fwd_y2_1d   = get_fwd_return(y2,           date, 1)
        fwd_y2_5d   = get_fwd_return(y2,           date, 5)
        fwd_sp_1d   = get_fwd_return(sp500,        date, 1)

        records.append({
            'date':        date,
            'event':       event_name,
            'value':       round(val, 3),
            'surprise_z':  round(z, 3) if not np.isnan(z) else np.nan,
            'surprise':    surp,
            'regime':      regime,
            'confidence':  conf,
            # Forward returns (bps for yields, pts for SP500)
            'y10_1d':      round(fwd_y10_1d  * 100, 2) if not np.isnan(fwd_y10_1d)  else np.nan,
            'y10_5d':      round(fwd_y10_5d  * 100, 2) if not np.isnan(fwd_y10_5d)  else np.nan,
            'y10_21d':     round(fwd_y10_21d * 100, 2) if not np.isnan(fwd_y10_21d) else np.nan,
            'curve_1d':    round(fwd_2s10_1d * 100, 2) if not np.isnan(fwd_2s10_1d) else np.nan,
            'curve_5d':    round(fwd_2s10_5d * 100, 2) if not np.isnan(fwd_2s10_5d) else np.nan,
            'y2_1d':       round(fwd_y2_1d   * 100, 2) if not np.isnan(fwd_y2_1d)   else np.nan,
            'y2_5d':       round(fwd_y2_5d   * 100, 2) if not np.isnan(fwd_y2_5d)   else np.nan,
            'sp500_1d':    round(fwd_sp_1d,   2)        if not np.isnan(fwd_sp_1d)   else np.nan,
        })

df = pd.DataFrame(records)
df.to_csv(OUTPUT_DIR + 'event_backtest_raw.csv', index=False)
print(f"Event table: {len(df)} rows across {df['event'].nunique()} events\n")

# ── Analysis ──────────────────────────────────────────────────────────────────

REGIME_ORDER = ['GOLDILOCKS','REFLATION','STAGFLATION_RISK','DEFLATION_RISK']
REGIME_SHORT = {
    'GOLDILOCKS':       'Goldilocks',
    'REFLATION':        'Reflation',
    'STAGFLATION_RISK': 'Stagfl. Risk',
    'DEFLATION_RISK':   'Deflation Risk',
}
REGIME_COLORS = {
    'GOLDILOCKS':       '#2ecc71',
    'REFLATION':        '#f39c12',
    'STAGFLATION_RISK': '#e74c3c',
    'DEFLATION_RISK':   '#3498db',
}

def sharpe(series, annualise=12):
    s = series.dropna()
    if len(s) < 3 or s.std() == 0:
        return np.nan
    return (s.mean() / s.std()) * np.sqrt(annualise)

def hit_rate(series):
    s = series.dropna()
    if len(s) == 0:
        return np.nan
    return (s > 0).mean()

# ── Table 1: Event × Regime × Surprise → mean 10Y yield change (bps) 5D ─────
print("=" * 70)
print("TABLE 1: Mean 10Y Yield Change (bps) — 5-Day Window")
print("BEAT = hot data above trend; MISS = cool data below trend")
print("Yield UP (positive) = bond prices fall = short ZN profitable")
print("=" * 70)

summary_rows = []
for event in df['event'].unique():
    for regime in REGIME_ORDER:
        for surp in ['BEAT', 'MISS']:
            mask = (df['event'] == event) & (df['regime'] == regime) & (df['surprise'] == surp)
            sub  = df[mask]['y10_5d'].dropna()
            if len(sub) < 3:
                continue
            # Trade direction: BEAT in inflation-heavy regimes → short ZN (yield goes up)
            # So we sign the P&L: BEAT → expect yield UP (+), MISS → expect yield DOWN (-)
            expected_sign = 1 if surp == 'BEAT' else -1
            signed_ret = sub * expected_sign  # positive = trade worked
            summary_rows.append({
                'event':      event,
                'regime':     REGIME_SHORT.get(regime, regime),
                'surprise':   surp,
                'n':          len(sub),
                'mean_bps':   round(sub.mean(), 1),
                'mean_signed':round(signed_ret.mean(), 1),
                'hit_rate':   round(hit_rate(signed_ret) * 100, 1),
                'sharpe':     round(sharpe(signed_ret), 2),
            })

summary = pd.DataFrame(summary_rows)

for event in summary['event'].unique():
    print(f"\n  {event}")
    print(f"  {'Regime':15} {'Surp':6} {'N':>4} {'Mean(bps)':>10} {'Signed':>8} {'Hit%':>7} {'Sharpe':>8}")
    print(f"  {'-'*60}")
    sub = summary[summary['event'] == event].sort_values(['regime','surprise'])
    for _, row in sub.iterrows():
        print(f"  {row['regime']:15} {row['surprise']:6} {int(row['n']):>4} "
              f"{row['mean_bps']:>10.1f} {row['mean_signed']:>8.1f} "
              f"{row['hit_rate']:>7.1f}% {row['sharpe']:>8.2f}")

# ── Table 2: Regime impact on CPI sensitivity ─────────────────────────────────
print("\n" + "=" * 70)
print("TABLE 2: CPI Signal Strength by Regime (5-day 10Y yield response)")
print("Demonstrates regime-conditioning of market sensitivity to CPI")
print("=" * 70)

cpi_df = df[df['event'] == 'CPI_MOM'].copy()
cpi_df['signed_ret'] = np.where(cpi_df['surprise'] == 'BEAT',
                                 cpi_df['y10_5d'],
                                -cpi_df['y10_5d'])
cpi_df = cpi_df[cpi_df['surprise'].isin(['BEAT','MISS'])]

print(f"\n  {'Regime':20} {'N':>4} {'Mean(bps)':>10} {'Hit%':>7} {'Sharpe':>8}")
print(f"  {'-'*52}")
for regime in REGIME_ORDER:
    sub = cpi_df[cpi_df['regime'] == regime]['signed_ret'].dropna()
    if len(sub) < 2:
        continue
    print(f"  {REGIME_SHORT.get(regime,regime):20} {len(sub):>4} "
          f"{sub.mean():>10.1f} {hit_rate(sub)*100:>7.1f}% {sharpe(sub):>8.2f}")

# ── Table 3: NFP sensitivity by regime ───────────────────────────────────────
print("\n" + "=" * 70)
print("TABLE 3: NFP Signal Strength by Regime (5-day 10Y yield response)")
print("=" * 70)

nfp_df = df[df['event'] == 'NFP'].copy()
nfp_df['signed_ret'] = np.where(nfp_df['surprise'] == 'BEAT',
                                 nfp_df['y10_5d'],
                                -nfp_df['y10_5d'])
nfp_df = nfp_df[nfp_df['surprise'].isin(['BEAT','MISS'])]

print(f"\n  {'Regime':20} {'N':>4} {'Mean(bps)':>10} {'Hit%':>7} {'Sharpe':>8}")
print(f"  {'-'*52}")
for regime in REGIME_ORDER:
    sub = nfp_df[nfp_df['regime'] == regime]['signed_ret'].dropna()
    if len(sub) < 2:
        continue
    print(f"  {REGIME_SHORT.get(regime,regime):20} {len(sub):>4} "
          f"{sub.mean():>10.1f} {hit_rate(sub)*100:>7.1f}% {sharpe(sub):>8.2f}")

# ── Simulate cumulative P&L ───────────────────────────────────────────────────
print("\nSimulating strategy P&L...")

# Strategy: trade CPI + NFP events only, conditioned on regime
# Entry: on release date, hold 5 days
# Position sizing: +1 if BEAT (short ZN), -1 if MISS (long ZN)
# Regime filter: only trade if regime = STAGFLATION_RISK or REFLATION
# (where inflation/growth data has historically highest impact)

strat_events = ['CPI_MOM', 'NFP']
strat_regimes = ['STAGFLATION_RISK', 'REFLATION', 'GOLDILOCKS', 'DEFLATION_RISK']

# Build four variants:
# V1: trade all regimes, no filter
# V2: trade only STAGFLATION + REFLATION
# V3: trade only HIGH confidence regime readings
# V4: trade all, but size by confidence (HIGH=1, MEDIUM=0.6, LOW=0.3)

def build_strategy(df, event_filter, regime_filter, conf_filter=None, conf_sizing=False):
    trades = df[
        df['event'].isin(event_filter) &
        df['regime'].isin(regime_filter) &
        df['surprise'].isin(['BEAT','MISS'])
    ].copy()

    if conf_filter:
        trades = trades[trades['confidence'].isin(conf_filter)]

    trades = trades.sort_values('date')
    trades['direction'] = np.where(trades['surprise'] == 'BEAT', 1, -1)

    if conf_sizing:
        size_map = {'HIGH': 1.0, 'MEDIUM': 0.6, 'LOW': 0.3}
        trades['size'] = trades['confidence'].map(size_map).fillna(0.3)
    else:
        trades['size'] = 1.0

    trades['pnl'] = trades['direction'] * trades['y10_5d'] * trades['size']
    trades['cum_pnl'] = trades['pnl'].cumsum()
    return trades

v1 = build_strategy(df, strat_events, REGIME_ORDER)
v2 = build_strategy(df, strat_events, ['STAGFLATION_RISK','REFLATION'])
v3 = build_strategy(df, strat_events, REGIME_ORDER, conf_filter=['HIGH'])
v4 = build_strategy(df, strat_events, REGIME_ORDER, conf_sizing=True)

for name, strat in [('V1 All regimes', v1), ('V2 Stagfl+Reflation only', v2),
                     ('V3 HIGH confidence only', v3), ('V4 Conf-sized', v4)]:
    pnl = strat['pnl'].dropna()
    if len(pnl) == 0:
        continue
    total = pnl.sum()
    sr    = sharpe(pnl)
    hr    = hit_rate(pnl) * 100
    n     = len(pnl)
    print(f"  {name:30} N={n:>3}  Total={total:>+7.1f}bps  "
          f"Sharpe={sr:>5.2f}  Hit%={hr:>5.1f}%")

# ── CHARTS ────────────────────────────────────────────────────────────────────
print("\nGenerating charts...")

fig = plt.figure(figsize=(22, 18), facecolor='#0d1117')
gs_main = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35,
                             top=0.93, bottom=0.06, left=0.07, right=0.97)

def style(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor('#0d1117')
    ax.tick_params(colors='#8b949e', labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#30363d')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', color='#21262d', linewidth=0.6, linestyle='--')
    if title:
        ax.set_title(title, color='#e6edf3', fontsize=11, fontweight='bold', pad=8, loc='left')
    if xlabel:
        ax.set_xlabel(xlabel, color='#8b949e', fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color='#8b949e', fontsize=9)

# ── Chart 1: CPI sensitivity by regime (mean signed return 5D) ───────────────
ax1 = fig.add_subplot(gs_main[0, 0])
style(ax1, title='CPI Surprise: 10Y Yield Response by Regime (5-day, bps)',
      ylabel='Mean signed bps (+ = trade worked)')

cpi_regime_data = []
for regime in REGIME_ORDER:
    sub = cpi_df[cpi_df['regime'] == regime]['signed_ret'].dropna()
    if len(sub) >= 2:
        cpi_regime_data.append({
            'regime': REGIME_SHORT.get(regime, regime),
            'mean':   sub.mean(),
            'se':     sub.sem(),
            'color':  REGIME_COLORS[regime],
            'n':      len(sub)
        })

crd = pd.DataFrame(cpi_regime_data)
bars = ax1.bar(crd['regime'], crd['mean'],
               color=crd['color'], alpha=0.85, edgecolor='#30363d', linewidth=0.5)
ax1.errorbar(crd['regime'], crd['mean'], yerr=crd['se']*1.96,
             fmt='none', color='white', capsize=4, linewidth=1.2)
ax1.axhline(0, color='#8b949e', linewidth=0.8, linestyle='--')
for bar, row in zip(bars, crd.itertuples()):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'n={row.n}', ha='center', va='bottom', fontsize=8, color='#8b949e')

# ── Chart 2: NFP sensitivity by regime ───────────────────────────────────────
ax2 = fig.add_subplot(gs_main[0, 1])
style(ax2, title='NFP Surprise: 10Y Yield Response by Regime (5-day, bps)',
      ylabel='Mean signed bps (+ = trade worked)')

nfp_regime_data = []
for regime in REGIME_ORDER:
    sub = nfp_df[nfp_df['regime'] == regime]['signed_ret'].dropna()
    if len(sub) >= 2:
        nfp_regime_data.append({
            'regime': REGIME_SHORT.get(regime, regime),
            'mean':   sub.mean(),
            'se':     sub.sem(),
            'color':  REGIME_COLORS[regime],
            'n':      len(sub)
        })

nrd = pd.DataFrame(nfp_regime_data)
bars2 = ax2.bar(nrd['regime'], nrd['mean'],
                color=nrd['color'], alpha=0.85, edgecolor='#30363d', linewidth=0.5)
ax2.errorbar(nrd['regime'], nrd['mean'], yerr=nrd['se']*1.96,
             fmt='none', color='white', capsize=4, linewidth=1.2)
ax2.axhline(0, color='#8b949e', linewidth=0.8, linestyle='--')
for bar, row in zip(bars2, nrd.itertuples()):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'n={row.n}', ha='center', va='bottom', fontsize=8, color='#8b949e')

# ── Chart 3: Heatmap — event × regime mean signed return ─────────────────────
ax3 = fig.add_subplot(gs_main[1, :])
style(ax3, title='Event × Regime Heatmap — Mean Signed 10Y Yield Response (5D, bps)')

heat_data = []
events_list = df['event'].unique().tolist()
for event in events_list:
    row_vals = []
    for regime in REGIME_ORDER:
        sub_beat = df[(df['event']==event) & (df['regime']==regime) & (df['surprise']=='BEAT')]['y10_5d'].dropna()
        sub_miss = df[(df['event']==event) & (df['regime']==regime) & (df['surprise']=='MISS')]['y10_5d'].dropna()
        signed = pd.concat([sub_beat, -sub_miss])
        row_vals.append(signed.mean() if len(signed) >= 3 else np.nan)
    heat_data.append(row_vals)

heat_df = pd.DataFrame(heat_data,
                        index=events_list,
                        columns=[REGIME_SHORT[r] for r in REGIME_ORDER])

# Custom diverging colormap
cmap = LinearSegmentedColormap.from_list('macro',
    ['#3498db', '#1a1a2e', '#e74c3c'], N=256)

vmax = max(abs(heat_df.values[~np.isnan(heat_df.values)]).max(), 1)
im = ax3.imshow(heat_df.values, cmap=cmap, aspect='auto',
                vmin=-vmax, vmax=vmax)

ax3.set_xticks(range(len(REGIME_ORDER)))
ax3.set_xticklabels([REGIME_SHORT[r] for r in REGIME_ORDER],
                     color='#e6edf3', fontsize=10, fontweight='bold')
ax3.set_yticks(range(len(events_list)))
ax3.set_yticklabels(events_list, color='#e6edf3', fontsize=10)

for i in range(len(events_list)):
    for j in range(len(REGIME_ORDER)):
        val = heat_df.values[i, j]
        if not np.isnan(val):
            ax3.text(j, i, f'{val:+.1f}', ha='center', va='center',
                     fontsize=10, fontweight='bold',
                     color='white' if abs(val) > vmax * 0.3 else '#8b949e')

plt.colorbar(im, ax=ax3, label='Mean signed bps', shrink=0.6,
             ).ax.yaxis.label.set_color('#8b949e')

# ── Chart 4: Cumulative P&L comparison ───────────────────────────────────────
ax4 = fig.add_subplot(gs_main[2, :])
style(ax4, title='Cumulative P&L — CPI + NFP Event Strategy (bps, 5-day ZN proxy)',
      ylabel='Cumulative bps', xlabel='Date')

strategy_styles = [
    (v1, '#8b949e', '--', 0.7, 'V1: All regimes, no filter'),
    (v2, '#f39c12', '-',  1.3, 'V2: Stagfl + Reflation only'),
    (v3, '#2ecc71', '-',  1.3, 'V3: HIGH confidence only'),
    (v4, '#58a6ff', '-',  1.5, 'V4: Confidence-sized'),
]

for strat, color, ls, lw, label in strategy_styles:
    s = strat.dropna(subset=['pnl']).set_index('date')['cum_pnl']
    if len(s) > 0:
        ax4.plot(s.index, s.values, color=color, linestyle=ls,
                 linewidth=lw, label=label, alpha=0.9)

ax4.axhline(0, color='#30363d', linewidth=0.8)
ax4.legend(loc='upper left', framealpha=0.3, fontsize=9,
           facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3')

import matplotlib.dates as mdates
ax4.xaxis.set_major_locator(mdates.YearLocator(2))
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax4.tick_params(axis='x', colors='#8b949e')

fig.suptitle('MACRO REGIME EVENT STRATEGY — FRED Proxy Backtest  |  2005–2026',
             color='#e6edf3', fontsize=14, fontweight='bold', y=0.97)
fig.text(0.07, 0.025,
         'Note: Surprise = z-score vs 12M rolling median (proxy for consensus). '
         'Forward returns = 10Y Treasury yield change × direction (bps). '
         'Bloomberg consensus data will improve signal quality significantly.',
         fontsize=8.5, color='#8b949e', ha='left')

plt.savefig(OUTPUT_DIR + 'event_backtest_chart.png', dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
print(f"Chart saved.")

df.to_csv(OUTPUT_DIR + 'event_backtest_raw.csv', index=False)
summary.to_csv(OUTPUT_DIR + 'event_backtest_summary.csv', index=False)
print("All files saved.")
