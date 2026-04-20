"""
Macro Regime Dashboard — Plotly Dash
=====================================
Run:  python dashboard.py
Open: http://127.0.0.1:8050

Features:
  - Date picker: select any date from 2000 to today
  - Regime label + confidence badge + transition warning
  - Growth score / inflation score gauges
  - 8 key metric cards with signal coloring
  - Growth score vs inflation score history chart (regime-shaded)
  - Regime history timeline
  - All data pulled live from FRED on load (cached for 24h)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from datetime import datetime, date
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

# ── Import our classifier ─────────────────────────────────────────────────────
from regime_classifier import MacroDataLoader, RegimeClassifier, FRED_API_KEY

# ── Pre-load data once at startup ─────────────────────────────────────────────
print("Loading FRED data... (this takes ~30 seconds on first run)")
loader = MacroDataLoader(api_key=FRED_API_KEY, start='1999-01-01')
loader.load_all()
classifier = RegimeClassifier(loader)

# Pre-compute full regime history for timeline chart
regime_history = pd.read_csv(
    os.path.join(os.path.dirname(__file__), 'regime_history.csv'),
    index_col='date', parse_dates=True
)
print("Ready.\n")

# ── Design tokens ─────────────────────────────────────────────────────────────
BG        = '#0d1117'
SURFACE   = '#161b22'
SURFACE2  = '#21262d'
BORDER    = '#30363d'
TEXT      = '#e6edf3'
TEXT_MUTED= '#8b949e'
TEXT_FAINT= '#484f58'

REGIME_COLORS = {
    'GOLDILOCKS':       '#2ecc71',
    'REFLATION':        '#f39c12',
    'STAGFLATION_RISK': '#e74c3c',
    'DEFLATION_RISK':   '#58a6ff',
}
REGIME_LABELS = {
    'GOLDILOCKS':       'GOLDILOCKS',
    'REFLATION':        'REFLATION',
    'STAGFLATION_RISK': 'STAGFLATION RISK',
    'DEFLATION_RISK':   'DEFLATION RISK',
}
REGIME_DESCRIPTIONS = {
    'GOLDILOCKS':       'Growth above trend · Inflation at/below target · Risk-on posture',
    'REFLATION':        'Growth accelerating · Inflation rising · Short duration · Long carry FX',
    'STAGFLATION_RISK': 'Growth decelerating · Inflation sticky · Short front-end · Safe-haven FX',
    'DEFLATION_RISK':   'Growth contracting · Inflation falling · Max long duration · Long USD/JPY',
}

STRATEGY_MAP = {
    'GOLDILOCKS':       [('Rates', 'Neutral duration · Steepener bias (2s10s)'),
                         ('FX',    'Long carry (AUD) · Short JPY/CHF'),
                         ('Risk',  'Full size · Risk-on')],
    'REFLATION':        [('Rates', 'Short duration · Long breakevens (TIPS vs ZN)'),
                         ('FX',    'Long commodity FX · Short USD'),
                         ('Risk',  'Full size · Steepener')],
    'STAGFLATION_RISK': [('Rates', 'Short front-end (ZT) · Flattener · Long TIPS'),
                         ('FX',    'Short high-beta FX · Long CHF/JPY'),
                         ('Risk',  'Half size · Hedged')],
    'DEFLATION_RISK':   [('Rates', 'Max long duration (ZB) · Bull flattener'),
                         ('FX',    'Long USD/JPY · Short carry basket'),
                         ('Risk',  'Minimum size · Safe haven only')],
}

# ── Helper: build gauge figure ─────────────────────────────────────────────────
def make_gauge(value, title, color):
    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=value,
        number={'font': {'size': 28, 'color': color}, 'suffix': ''},
        title={'text': title, 'font': {'size': 13, 'color': TEXT_MUTED}},
        gauge={
            'axis': {'range': [-1, 1], 'tickwidth': 1,
                     'tickcolor': BORDER, 'tickfont': {'color': TEXT_MUTED, 'size': 10},
                     'dtick': 0.5},
            'bar': {'color': color, 'thickness': 0.25},
            'bgcolor': SURFACE2,
            'borderwidth': 0,
            'steps': [
                {'range': [-1, -0.5], 'color': '#1a1f28'},
                {'range': [-0.5, 0],  'color': '#1d2230'},
                {'range': [0, 0.5],   'color': '#1d2a24'},
                {'range': [0.5, 1],   'color': '#1a2820'},
            ],
            'threshold': {
                'line': {'color': 'white', 'width': 2},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': TEXT},
    )
    return fig

# ── Helper: regime history chart ──────────────────────────────────────────────
def make_history_chart(selected_date):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.4, 0.6], vertical_spacing=0.04)

    # Shade regimes
    prev_regime = None
    start_dt = None
    for dt, row in regime_history.iterrows():
        regime = row['regime']
        if regime != prev_regime:
            if prev_regime is not None:
                fig.add_vrect(
                    x0=start_dt, x1=dt,
                    fillcolor=REGIME_COLORS.get(prev_regime, '#555'),
                    opacity=0.12, layer='below', line_width=0,
                    row=1, col=1
                )
                fig.add_vrect(
                    x0=start_dt, x1=dt,
                    fillcolor=REGIME_COLORS.get(prev_regime, '#555'),
                    opacity=0.12, layer='below', line_width=0,
                    row=2, col=1
                )
            start_dt = dt
            prev_regime = regime
    if prev_regime is not None:
        fig.add_vrect(
            x0=start_dt, x1=regime_history.index[-1],
            fillcolor=REGIME_COLORS.get(prev_regime, '#555'),
            opacity=0.12, layer='below', line_width=0,
            row=1, col=1
        )
        fig.add_vrect(
            x0=start_dt, x1=regime_history.index[-1],
            fillcolor=REGIME_COLORS.get(prev_regime, '#555'),
            opacity=0.12, layer='below', line_width=0,
            row=2, col=1
        )

    # Row 1: Growth + Inflation scores
    fig.add_trace(go.Scatter(
        x=regime_history.index, y=regime_history['growth_score'],
        name='Growth Score', line=dict(color='#2ecc71', width=1.5),
        hovertemplate='%{x|%b %Y}: %{y:.2f}<extra>Growth</extra>'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=regime_history.index, y=regime_history['inflation_score'],
        name='Inflation Score', line=dict(color='#e74c3c', width=1.5),
        hovertemplate='%{x|%b %Y}: %{y:.2f}<extra>Inflation</extra>'
    ), row=1, col=1)
    fig.add_hline(y=0, line_color=BORDER, line_width=1, row=1, col=1)

    # Row 2: Regime as numeric with color
    regime_num = regime_history['regime'].map({
        'DEFLATION_RISK': 1, 'GOLDILOCKS': 2,
        'REFLATION': 3, 'STAGFLATION_RISK': 4
    })
    for regime, num in [('DEFLATION_RISK',1),('GOLDILOCKS',2),
                         ('REFLATION',3),('STAGFLATION_RISK',4)]:
        mask = regime_history['regime'] == regime
        fig.add_trace(go.Scatter(
            x=regime_history.index[mask],
            y=regime_num[mask],
            mode='markers',
            name=REGIME_LABELS[regime],
            marker=dict(color=REGIME_COLORS[regime], size=6, symbol='circle'),
            showlegend=False,
            hovertemplate=f'%{{x|%b %Y}}: {REGIME_LABELS[regime]}<extra></extra>'
        ), row=2, col=1)

    # Selected date marker
    sel_ts = pd.Timestamp(selected_date)
    if sel_ts in regime_history.index:
        sel_regime = regime_history.loc[sel_ts, 'regime']
        sel_num = {'DEFLATION_RISK':1,'GOLDILOCKS':2,'REFLATION':3,'STAGFLATION_RISK':4}.get(sel_regime,2)
        fig.add_vline(x=sel_ts, line_color='white', line_width=1.5,
                      line_dash='dot', row=1, col=1)
        fig.add_vline(x=sel_ts, line_color='white', line_width=1.5,
                      line_dash='dot', row=2, col=1)
        fig.add_trace(go.Scatter(
            x=[sel_ts], y=[sel_num],
            mode='markers',
            marker=dict(color='white', size=12, symbol='diamond',
                        line=dict(color=REGIME_COLORS.get(sel_regime,'white'), width=2)),
            name='Selected date',
            hovertemplate=f'{sel_ts.strftime("%b %Y")}: {REGIME_LABELS.get(sel_regime,"")}<extra>Selected</extra>'
        ), row=2, col=1)

    fig.update_yaxes(
        row=2, tickvals=[1,2,3,4],
        ticktext=['Defl. Risk','Goldilocks','Reflation','Stagfl. Risk'],
        tickfont=dict(size=10, color=TEXT_MUTED),
        gridcolor=BORDER
    )
    fig.update_yaxes(row=1, title_text='Score', title_font=dict(size=11, color=TEXT_MUTED),
                     range=[-1.1,1.1], gridcolor=BORDER, tickfont=dict(color=TEXT_MUTED))
    fig.update_xaxes(gridcolor=BORDER, tickfont=dict(color=TEXT_MUTED))

    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': TEXT, 'size': 11},
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0,
            font=dict(size=10, color=TEXT_MUTED),
            bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)'
        ),
        hovermode='x unified',
    )
    return fig

# ── Metric card component ──────────────────────────────────────────────────────
def metric_card(label, value, signal, delta=None):
    signal_colors = {
        'BULLISH':  '#2ecc71',
        'BEARISH':  '#e74c3c',
        'NEUTRAL':  TEXT_MUTED,
        'WARNING':  '#f39c12',
        'N/A':      TEXT_FAINT,
    }
    color = signal_colors.get(signal, TEXT_MUTED)
    delta_el = html.Span(
        f' {delta}', style={'fontSize': '11px', 'color': TEXT_MUTED}
    ) if delta else None

    return html.Div([
        html.Div(label, style={
            'fontSize': '11px', 'color': TEXT_MUTED,
            'textTransform': 'uppercase', 'letterSpacing': '0.08em',
            'marginBottom': '6px'
        }),
        html.Div([
            html.Span(str(value), style={
                'fontSize': '22px', 'fontWeight': '700', 'color': color
            }),
            delta_el or '',
        ]),
        html.Div(signal, style={
            'fontSize': '10px', 'color': color, 'marginTop': '4px',
            'fontWeight': '600', 'letterSpacing': '0.05em'
        }),
    ], style={
        'background': SURFACE,
        'border': f'1px solid {BORDER}',
        'borderRadius': '8px',
        'padding': '16px',
    })

# ── App layout ────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title='Macro Regime Terminal'
)

app.layout = html.Div(style={
    'backgroundColor': BG,
    'minHeight': '100vh',
    'fontFamily': '"Inter", "Segoe UI", system-ui, sans-serif',
    'color': TEXT,
    'padding': '0',
}, children=[

    # ── Header ────────────────────────────────────────────────────────────────
    html.Div(style={
        'background': SURFACE,
        'borderBottom': f'1px solid {BORDER}',
        'padding': '16px 32px',
        'display': 'flex',
        'alignItems': 'center',
        'justifyContent': 'space-between',
        'gap': '24px',
    }, children=[
        html.Div([
            html.Span('◈ ', style={'color': '#58a6ff', 'fontSize': '18px'}),
            html.Span('MACRO REGIME TERMINAL', style={
                'fontSize': '14px', 'fontWeight': '700',
                'letterSpacing': '0.12em', 'color': TEXT
            }),
            html.Span(' v1.0', style={
                'fontSize': '11px', 'color': TEXT_MUTED,
                'marginLeft': '8px'
            }),
        ]),
        html.Div([
            html.Span('DATE ', style={
                'fontSize': '11px', 'color': TEXT_MUTED,
                'letterSpacing': '0.08em', 'marginRight': '10px'
            }),
            dcc.DatePickerSingle(
                id='date-picker',
                min_date_allowed=date(2000, 1, 1),
                max_date_allowed=date.today(),
                initial_visible_month=date.today(),
                date=str(date.today()),
                display_format='MMM DD, YYYY',
                style={'fontSize': '13px'},
                className='dash-datepicker-dark',
            ),
        ], style={'display': 'flex', 'alignItems': 'center'}),
        html.Div([
            html.Span('FRED · BLS · BEA · TREASURY', style={
                'fontSize': '10px', 'color': TEXT_FAINT,
                'letterSpacing': '0.08em'
            }),
        ]),
    ]),

    # ── Main content ──────────────────────────────────────────────────────────
    html.Div(style={'padding': '24px 32px', 'maxWidth': '1400px', 'margin': '0 auto'},
    children=[

        # ── Row 1: Regime banner ───────────────────────────────────────────
        html.Div(id='regime-banner', style={'marginBottom': '24px'}),

        # ── Row 2: Gauges + Metrics ────────────────────────────────────────
        html.Div(style={'display': 'flex', 'gap': '20px', 'marginBottom': '24px',
                        'flexWrap': 'wrap'}, children=[
            # Growth gauge
            html.Div([
                dcc.Graph(id='growth-gauge',
                          config={'displayModeBar': False},
                          style={'height': '200px'}),
            ], style={
                'background': SURFACE, 'border': f'1px solid {BORDER}',
                'borderRadius': '8px', 'padding': '8px',
                'flex': '0 0 200px',
            }),
            # Inflation gauge
            html.Div([
                dcc.Graph(id='inflation-gauge',
                          config={'displayModeBar': False},
                          style={'height': '200px'}),
            ], style={
                'background': SURFACE, 'border': f'1px solid {BORDER}',
                'borderRadius': '8px', 'padding': '8px',
                'flex': '0 0 200px',
            }),
            # Metric cards
            html.Div(id='metric-cards', style={
                'flex': '1', 'minWidth': '0',
            }),
        ]),

        # ── Row 3: Strategy box ────────────────────────────────────────────
        html.Div(id='strategy-box', style={'marginBottom': '24px'}),

        # ── Row 4: History chart ───────────────────────────────────────────
        html.Div([
            html.Div('REGIME HISTORY  2000 – TODAY', style={
                'fontSize': '11px', 'color': TEXT_MUTED,
                'letterSpacing': '0.1em', 'padding': '16px 20px 8px',
                'fontWeight': '600'
            }),
            dcc.Graph(id='history-chart',
                      config={'displayModeBar': False}),
        ], style={
            'background': SURFACE, 'border': f'1px solid {BORDER}',
            'borderRadius': '8px',
        }),

        # ── Footer ────────────────────────────────────────────────────────
        html.Div(
            'Data: FRED API · Regime = rules-based classifier (CFNAI, Core PCE, CPI, GDP, credit spreads, VIX) · '
            'White ring = HIGH confidence · No look-ahead bias',
            style={'fontSize': '11px', 'color': TEXT_FAINT,
                   'marginTop': '20px', 'textAlign': 'center',
                   'letterSpacing': '0.04em'}
        ),
    ]),

])

# Inject dark datepicker CSS via assets
app.index_string = '''
<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<style>
.dash-datepicker-dark .DateInput_input {
    background: #21262d !important; color: #e6edf3 !important;
    border: 1px solid #30363d !important; border-radius: 6px;
    font-size: 13px; padding: 6px 10px;
}
.dash-datepicker-dark .DateInput_input:focus { border-color: #58a6ff !important; outline: none; }
.SingleDatePickerInput { background: transparent !important; border: none !important; }
.DateRangePicker_picker, .SingleDatePicker_picker {
    background: #161b22 !important; border: 1px solid #30363d !important; border-radius: 8px !important;
}
.CalendarDay__default { background: #161b22 !important; color: #e6edf3 !important; border: 1px solid #21262d !important; }
.CalendarDay__selected { background: #58a6ff !important; color: #0d1117 !important; }
.CalendarDay__default:hover { background: #30363d !important; }
.DayPickerNavigation_button { border: 1px solid #30363d !important; background: #21262d !important; }
.DayPickerNavigation_svg__horizontal { fill: #e6edf3 !important; }
.CalendarMonth_caption { color: #e6edf3 !important; }
.DayPicker_weekHeader_li small { color: #8b949e !important; }
.DateInput { background: transparent !important; }
body { margin: 0; }
</style>
</head>
<body>
{%app_entry%}
<footer>
{%config%}
{%scripts%}
{%renderer%}
</footer>
</body>
</html>
'''

# ── Callbacks ──────────────────────────────────────────────────────────────────
@app.callback(
    Output('regime-banner',   'children'),
    Output('growth-gauge',    'figure'),
    Output('inflation-gauge', 'figure'),
    Output('metric-cards',    'children'),
    Output('strategy-box',    'children'),
    Output('history-chart',   'figure'),
    Input('date-picker',      'date'),
)
def update_dashboard(selected_date):
    if not selected_date:
        selected_date = str(date.today())

    sel_dt = pd.Timestamp(selected_date)

    # ── Run classifier ─────────────────────────────────────────────────────
    result = classifier.classify(sel_dt)
    regime     = result['regime']
    growth     = result['growth_score']
    inflation  = result['inflation_score']
    confidence = result['confidence']
    warning    = result['transition_warning']
    signals    = result['signals']

    regime_color = REGIME_COLORS.get(regime, TEXT_MUTED)
    regime_label = REGIME_LABELS.get(regime, regime)
    regime_desc  = REGIME_DESCRIPTIONS.get(regime, '')

    # ── Regime banner ──────────────────────────────────────────────────────
    conf_bg = {'HIGH': '#1a2820', 'MEDIUM': '#2a2210', 'LOW': '#1a1a28'}.get(confidence, SURFACE2)
    conf_color = {'HIGH': '#2ecc71', 'MEDIUM': '#f39c12', 'LOW': '#58a6ff'}.get(confidence, TEXT_MUTED)

    warning_badge = html.Span(
        '⚡ TRANSITION WARNING', style={
            'background': '#3d1f1f', 'color': '#e74c3c',
            'fontSize': '10px', 'fontWeight': '700',
            'letterSpacing': '0.1em', 'padding': '3px 10px',
            'borderRadius': '4px', 'border': '1px solid #e74c3c',
            'marginLeft': '12px',
        }
    ) if warning else ''

    banner = html.Div(style={
        'background': SURFACE,
        'border': f'2px solid {regime_color}',
        'borderRadius': '10px',
        'padding': '20px 28px',
        'display': 'flex',
        'alignItems': 'center',
        'justifyContent': 'space-between',
        'flexWrap': 'wrap',
        'gap': '16px',
    }, children=[
        html.Div([
            html.Div([
                html.Span(regime_label, style={
                    'fontSize': '22px', 'fontWeight': '800',
                    'color': regime_color, 'letterSpacing': '0.05em',
                }),
                html.Span(f' — {sel_dt.strftime("%B %d, %Y")}', style={
                    'fontSize': '14px', 'color': TEXT_MUTED,
                    'marginLeft': '12px',
                }),
                warning_badge,
            ], style={'display': 'flex', 'alignItems': 'center', 'flexWrap': 'wrap', 'gap': '4px'}),
            html.Div(regime_desc, style={
                'fontSize': '13px', 'color': TEXT_MUTED,
                'marginTop': '6px', 'letterSpacing': '0.02em',
            }),
        ]),
        html.Div([
            html.Div('CONFIDENCE', style={
                'fontSize': '10px', 'color': TEXT_FAINT,
                'letterSpacing': '0.1em', 'marginBottom': '4px',
            }),
            html.Div(confidence, style={
                'fontSize': '18px', 'fontWeight': '700',
                'color': conf_color,
                'background': conf_bg,
                'padding': '6px 18px',
                'borderRadius': '6px',
                'border': f'1px solid {conf_color}',
                'letterSpacing': '0.08em',
            }),
        ]),
    ])

    # ── Gauges ────────────────────────────────────────────────────────────
    growth_color = '#2ecc71' if growth >= 0 else '#e74c3c'
    infl_color   = '#e74c3c' if inflation >= 0 else '#58a6ff'

    growth_fig = make_gauge(growth,    'GROWTH SCORE',    growth_color)
    infl_fig   = make_gauge(inflation, 'INFLATION SCORE', infl_color)

    # ── Metric cards ──────────────────────────────────────────────────────
    def sig(val, high_is_good=True, threshold_high=None, threshold_low=None):
        """Return signal string based on value vs thresholds."""
        if val is None:
            return 'N/A'
        if threshold_high is not None and threshold_low is not None:
            if val > threshold_high:
                return 'BULLISH' if high_is_good else 'BEARISH'
            elif val < threshold_low:
                return 'BEARISH' if high_is_good else 'BULLISH'
            return 'NEUTRAL'
        return 'NEUTRAL'

    core_pce   = signals.get('core_pce_yoy')
    core_cpi   = signals.get('core_cpi_yoy')
    unemp      = signals.get('unemployment')
    cfnai      = signals.get('cfnai')
    credit     = signals.get('credit_spread')
    vix        = signals.get('vix')
    spread2s10 = signals.get('spread_2s10s')
    breakeven  = signals.get('breakeven_10y')

    def pce_sig(v):
        if v is None: return 'N/A'
        if v > 3.0:  return 'BEARISH'
        if v < 2.0:  return 'BULLISH'
        return 'NEUTRAL'

    def unemp_sig(v):
        if v is None: return 'N/A'
        if v > 5.0:  return 'BEARISH'
        if v < 4.0:  return 'BULLISH'
        return 'NEUTRAL'

    def credit_sig(v):
        if v is None: return 'N/A'
        if v > 5.0:  return 'BEARISH'
        if v < 3.5:  return 'BULLISH'
        return 'NEUTRAL'

    def vix_sig(v):
        if v is None: return 'N/A'
        if v > 30:   return 'BEARISH'
        if v < 18:   return 'BULLISH'
        return 'NEUTRAL'

    def curve_sig(v):
        if v is None: return 'N/A'
        if v < 0:    return 'BEARISH'   # inverted
        if v > 50:   return 'BULLISH'
        return 'NEUTRAL'

    def cfnai_sig(v):
        if v is None: return 'N/A'
        if v > 0.2:  return 'BULLISH'
        if v < -0.5: return 'BEARISH'
        return 'NEUTRAL'

    cards = html.Div(style={
        'display': 'grid',
        'gridTemplateColumns': 'repeat(4, 1fr)',
        'gap': '12px',
        'width': '100%',
    }, children=[
        metric_card('Core PCE YoY',   f'{core_pce:.1f}%'  if core_pce  else 'N/A', pce_sig(core_pce)),
        metric_card('Core CPI YoY',   f'{core_cpi:.1f}%'  if core_cpi  else 'N/A', pce_sig(core_cpi)),
        metric_card('Unemployment',   f'{unemp:.1f}%'     if unemp     else 'N/A', unemp_sig(unemp)),
        metric_card('CFNAI (3M MA)',  f'{cfnai:.2f}'      if cfnai     else 'N/A', cfnai_sig(cfnai)),
        metric_card('HY Spread',      f'{credit:.2f}%'    if credit    else 'N/A', credit_sig(credit)),
        metric_card('VIX',            f'{vix:.1f}'        if vix       else 'N/A', vix_sig(vix)),
        metric_card('2s10s Spread',   f'{spread2s10:+.0f}bps' if spread2s10 else 'N/A', curve_sig(spread2s10)),
        metric_card('10Y Breakeven',  f'{breakeven:.2f}%' if breakeven else 'N/A',
                    'BEARISH' if (breakeven or 0) > 2.8 else 'BULLISH' if (breakeven or 0) < 2.0 else 'NEUTRAL'),
    ])

    # ── Strategy box ──────────────────────────────────────────────────────
    strat_rows = STRATEGY_MAP.get(regime, [])
    strat_items = []
    for category, action in strat_rows:
        strat_items.append(html.Div(style={
            'display': 'flex', 'gap': '16px', 'alignItems': 'flex-start',
            'padding': '10px 0',
            'borderBottom': f'1px solid {BORDER}',
        }, children=[
            html.Span(category.upper(), style={
                'fontSize': '10px', 'fontWeight': '700',
                'color': regime_color, 'letterSpacing': '0.1em',
                'minWidth': '60px', 'paddingTop': '2px',
            }),
            html.Span(action, style={
                'fontSize': '13px', 'color': TEXT,
            }),
        ]))

    strategy_box = html.Div([
        html.Div('STRATEGY POSTURE', style={
            'fontSize': '11px', 'color': TEXT_MUTED,
            'letterSpacing': '0.1em', 'fontWeight': '600',
            'marginBottom': '4px',
        }),
        html.Div(strat_items),
    ], style={
        'background': SURFACE,
        'border': f'1px solid {BORDER}',
        'borderLeft': f'3px solid {regime_color}',
        'borderRadius': '8px',
        'padding': '16px 20px',
    })

    # ── History chart ──────────────────────────────────────────────────────
    history_fig = make_history_chart(sel_dt)

    return banner, growth_fig, infl_fig, cards, strategy_box, history_fig


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*50)
    print("  MACRO REGIME TERMINAL")
    print("  http://127.0.0.1:8050")
    print("="*50 + "\n")
    app.run(debug=False, port=8050)
