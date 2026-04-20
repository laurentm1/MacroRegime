"""
Macro Regime — Visualisation
Generates a publication-quality chart of regime history 2000–today
overlaid with S&P 500 and key macro events.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from fredapi import Fred
import warnings
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
FRED_API_KEY = 'b1116b7ae7351cdf8018805fbc612ec3'
CSV_PATH     = '/home/user/workspace/macro_regime/regime_history.csv'
OUTPUT_PATH  = '/home/user/workspace/macro_regime/regime_chart.png'

REGIME_COLORS = {
    'GOLDILOCKS':       '#2ecc71',   # green
    'REFLATION':        '#f39c12',   # amber
    'STAGFLATION_RISK': '#e74c3c',   # red
    'DEFLATION_RISK':   '#3498db',   # blue
}

REGIME_LABELS = {
    'GOLDILOCKS':       'Goldilocks',
    'REFLATION':        'Reflation',
    'STAGFLATION_RISK': 'Stagflation Risk',
    'DEFLATION_RISK':   'Deflation Risk',
}

KEY_EVENTS = [
    ('2001-09-11', '9/11',            'top'),
    ('2008-09-15', 'Lehman',          'top'),
    ('2010-05-01', 'EU Debt Crisis',  'bottom'),
    ('2013-05-22', 'Taper Tantrum',   'top'),
    ('2020-03-01', 'COVID',           'top'),
    ('2022-03-01', 'Fed Hikes Begin', 'bottom'),
    ('2023-03-01', 'SVB Crisis',      'top'),
]

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH, index_col='date', parse_dates=True)

# Pull SP500 from FRED
fred = Fred(api_key=FRED_API_KEY)
sp500 = fred.get_series('SP500', observation_start='2000-01-01')
sp500 = sp500.resample('MS').last().reindex(df.index, method='nearest')

# Pull 10Y yield
y10 = fred.get_series('GS10', observation_start='2000-01-01')
y10 = y10.resample('MS').last().reindex(df.index, method='nearest')

# Pull Core PCE YoY
core_pce_raw = fred.get_series('PCEPILFE', observation_start='2000-01-01')
core_pce_yoy = core_pce_raw.pct_change(12) * 100
core_pce_yoy = core_pce_yoy.resample('MS').last().reindex(df.index, method='nearest')

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 16), facecolor='#0d1117')
gs  = GridSpec(4, 1, figure=fig, hspace=0.10,
               top=0.93, bottom=0.07, left=0.08, right=0.97)

axes = [fig.add_subplot(gs[i]) for i in range(4)]

def style_ax(ax):
    ax.set_facecolor('#0d1117')
    ax.tick_params(colors='#8b949e', labelsize=9)
    ax.spines['bottom'].set_color('#30363d')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#30363d')
    ax.yaxis.label.set_color('#8b949e')
    ax.grid(axis='y', color='#21262d', linewidth=0.6, linestyle='--')
    ax.grid(axis='x', color='#21262d', linewidth=0.3)

for ax in axes:
    style_ax(ax)

def shade_regimes(ax, df):
    prev_regime = None
    start_date  = None
    for date, row in df.iterrows():
        regime = row['regime']
        if regime != prev_regime:
            if prev_regime is not None:
                ax.axvspan(start_date, date,
                           color=REGIME_COLORS.get(prev_regime, '#555'),
                           alpha=0.18, linewidth=0)
            start_date  = date
            prev_regime = regime
    if prev_regime is not None:
        ax.axvspan(start_date, df.index[-1],
                   color=REGIME_COLORS.get(prev_regime, '#555'),
                   alpha=0.18, linewidth=0)

# ── Panel 1: Regime timeline ──────────────────────────────────────────────────
ax1 = axes[0]
shade_regimes(ax1, df)

regime_numeric = df['regime'].map({
    'GOLDILOCKS':       2,
    'REFLATION':        3,
    'STAGFLATION_RISK': 4,
    'DEFLATION_RISK':   1,
})

for regime, num in [('DEFLATION_RISK',1),('GOLDILOCKS',2),('REFLATION',3),('STAGFLATION_RISK',4)]:
    mask = df['regime'] == regime
    ax1.scatter(df.index[mask], regime_numeric[mask],
                color=REGIME_COLORS[regime], s=60, zorder=5, label=REGIME_LABELS[regime], alpha=0.9)

ax1.set_yticks([1, 2, 3, 4])
ax1.set_yticklabels(['Deflation Risk', 'Goldilocks', 'Reflation', 'Stagflation Risk'],
                     fontsize=10, color='#e6edf3', fontweight='bold')
ax1.set_ylabel('Regime', fontsize=10, color='#e6edf3')
ax1.set_xticklabels([])
ax1.set_title('MACRO REGIME CLASSIFIER  |  2000 – 2026  |  FRED Data',
              color='#e6edf3', fontsize=15, fontweight='bold', loc='left', pad=12)

# Confidence overlay
high_conf  = df[df['confidence'] == 'HIGH'].index
ax1.scatter(high_conf, regime_numeric[high_conf],
            edgecolors='white', facecolors='none', s=110, linewidths=1.2, zorder=6)
ax1.scatter(df[df['transition_warning']].index,
            regime_numeric[df['transition_warning']],
            marker='^', color='#ff6b6b', s=110, zorder=7, label='Transition Warning')

legend = ax1.legend(loc='lower right', framealpha=0.3, fontsize=8,
                    facecolor='#161b22', edgecolor='#30363d',
                    labelcolor='#e6edf3')

# ── Panel 2: S&P 500 ──────────────────────────────────────────────────────────
ax2 = axes[1]
shade_regimes(ax2, df)
ax2.plot(sp500.index, sp500.values, color='#58a6ff', linewidth=1.2, zorder=5)
ax2.set_ylabel('S&P 500', fontsize=10, color='#e6edf3', fontweight='bold')
ax2.set_yscale('log')
ax2.set_xticklabels([])
ax2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda x, _: f'{int(x):,}'))

# ── Panel 3: Growth & Inflation scores ───────────────────────────────────────
ax3 = axes[2]
shade_regimes(ax3, df)
ax3.plot(df.index, df['growth_score'],    color='#2ecc71', linewidth=1.2,
         label='Growth Score', zorder=5)
ax3.plot(df.index, df['inflation_score'], color='#e74c3c', linewidth=1.2,
         label='Inflation Score', zorder=5)
ax3.axhline(0, color='#8b949e', linewidth=0.8, linestyle='--', alpha=0.7)
ax3.set_ylabel('Score (-1 to +1)', fontsize=10, color='#e6edf3')
ax3.set_ylim(-1.1, 1.1)
ax3.set_xticklabels([])
ax3.legend(loc='upper right', framealpha=0.3, fontsize=8,
           facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3')

# ── Panel 4: Core PCE & 10Y Yield ────────────────────────────────────────────
ax4 = axes[3]
shade_regimes(ax4, df)
ax4_twin = ax4.twinx()
ax4_twin.set_facecolor('#0d1117')

ax4.plot(core_pce_yoy.index, core_pce_yoy.values,
         color='#ff7b72', linewidth=1.2, label='Core PCE YoY %', zorder=5)
ax4.axhline(2.5, color='#ff7b72', linewidth=0.7, linestyle=':', alpha=0.6)
ax4.set_ylabel('Core PCE YoY %', fontsize=10, color='#ff7b72', fontweight='bold')
ax4.tick_params(axis='y', colors='#ff7b72', labelsize=10)
ax4.spines['left'].set_color('#ff7b72')

ax4_twin.plot(y10.index, y10.values,
              color='#79c0ff', linewidth=1.2, label='10Y Treasury', zorder=5)
ax4_twin.set_ylabel('10Y Yield %', fontsize=10, color='#79c0ff', fontweight='bold')
ax4_twin.tick_params(axis='y', colors='#79c0ff', labelsize=10)
ax4_twin.spines['right'].set_color('#79c0ff')
ax4_twin.spines['top'].set_visible(False)
ax4_twin.spines['bottom'].set_color('#30363d')
ax4_twin.spines['left'].set_color('#30363d')
ax4_twin.grid(False)

# Combined legend
lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_twin.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2,
           loc='upper left', framealpha=0.3, fontsize=8,
           facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3')

# ── Event annotations (on ax2 = S&P panel) ────────────────────────────────────
for date_str, label, position in KEY_EVENTS:
    dt = pd.Timestamp(date_str)
    if dt < df.index[0] or dt > df.index[-1]:
        continue
    for ax in [ax1, ax2, ax3, ax4]:
        ax.axvline(dt, color='#8b949e', linewidth=0.6, linestyle=':', alpha=0.7, zorder=3)
    y_val = sp500.asof(dt) if dt in sp500.index or sp500.index[0] <= dt <= sp500.index[-1] else None
    if y_val and not np.isnan(y_val):
        offset = 1.20 if position == 'top' else 0.82
        ax2.annotate(label,
                     xy=(dt, y_val),
                     xytext=(dt, y_val * offset),
                     fontsize=7.5, color='#8b949e',
                     ha='center', va='bottom',
                     arrowprops=dict(arrowstyle='->', color='#8b949e',
                                     lw=0.8, connectionstyle='arc3,rad=0'))

# ── X-axis formatting ─────────────────────────────────────────────────────────
import matplotlib.dates as mdates
for ax in [ax4]:
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', colors='#8b949e', labelsize=9)

for ax in [ax1, ax2, ax3]:
    ax.set_xlim(df.index[0], df.index[-1])
ax4.set_xlim(df.index[0], df.index[-1])
ax4_twin.set_xlim(df.index[0], df.index[-1])

# ── Footer ────────────────────────────────────────────────────────────────────
fig.text(0.08, 0.025,
         'Source: FRED (Federal Reserve Bank of St. Louis)  ·  '
         'Regime = rules-based classifier: CFNAI, Core PCE, CPI, GDP, credit spreads, VIX  ·  '
         'White ring = HIGH confidence  ·  Red triangle = transition warning',
         fontsize=9, color='#8b949e', ha='left')

plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight',
            facecolor='#0d1117', edgecolor='none')
print(f"Chart saved to {OUTPUT_PATH}")
