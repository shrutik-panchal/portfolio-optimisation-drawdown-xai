#-------------------------------------------------------------------------------------
"""
app.py
------
Dash dashboard for the Portfolio Optimisation + Drawdown + SHAP XAI project.

Layout:-
  - Sidebar  : ticker input, date range, market selector, run button
  - Main     : 6 chart panels + strategy comparison table + SHAP panels

Panels:
  1. Efficient Frontier (Markowitz)
  2. CDaR Frontier
  3. Portfolio Weights (bar chart - all 3 strategies)
  4. Drawdown Underwater Curve
  5. SHAP Global Feature Importance
  6. SHAP Signed Direction Heatmap
  7. Strategy Comparison Table
  8. Per-Asset Drawdown Table
  9. Waterfall (user-selected ticker)
"""
#-------------------------------------------------------------------------------------
# Importing required libraries and modules
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
import sys
import os
import logging
import warnings
import numpy as np
import pandas as pd

import dash
from dash import dcc, html, Input, Output, State, dash_table, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
#-------------------------------------------------------------------------------------
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetch_data import get_data
from models.markowitz import run_markowitz
from models.drawdown import run_drawdown, compute_portfolio_drawdown
from models.shap_attribution import run_shap, waterfall_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)
#-------------------------------------------------------------------------------------
# Constants
#-------------------------------------------------------------------------------------
RISK_FREE_RATES = {
    "IN": 0.0525,   # RBI Repo Rate April 2026
    "US": 0.0430,   # Fed Funds Rate
    "EU": 0.0250,   # ECB Rate
}

MARKET_LABELS = {
    "IN": "🇮🇳 India (NSE/BSE)  -  RFR: 5.25%",
    "US": "🇺🇸 United States    -  RFR: 4.30%",
    "EU": "🇪🇺 Europe           -  RFR: 2.50%",
}

DEFAULT_TICKERS  = "AXISBANK.NS, HDFCBANK.NS, ICICIBANK.NS, KOTAKBANK.NS, PNB.NS, SBIN.NS"
DEFAULT_START    = "2015-01-01"
DEFAULT_END      = pd.Timestamp.today().strftime("%Y-%m-%d")
MIN_TRADING_DAYS = 756    # 3 years
WARN_TRADING_DAYS = 1260  # 5 years
#-------------------------------------------------------------------------------------
# Colour Palette
#-------------------------------------------------------------------------------------
COLORS = {
    "bg":         "#0f1117",
    "surface":    "#1a1d27",
    "surface2":   "#21253a",
    "border":     "#2e3347",
    "text":       "#e2e4ed",
    "muted":      "#8b90a0",
    "primary":    "#4f98a3",
    "green":      "#6daa45",
    "orange":     "#fdab43",
    "red":        "#dd6974",
    "purple":     "#a86fdf",
    "gold":       "#e8af34",
    "max_sharpe": "#4f98a3",
    "min_vol":    "#6daa45",
    "min_cdar":   "#fdab43",
    "equal_w":    "#a86fdf",
}

CHART_LAYOUT = dict(
    paper_bgcolor = COLORS["surface"],
    plot_bgcolor  = COLORS["surface"],
    font          = dict(color=COLORS["text"], family="Inter, sans-serif", size=12),
    margin        = dict(l=50, r=20, t=40, b=50),
)

AXIS_STYLE = dict(
    gridcolor     = COLORS["border"],
    zerolinecolor = COLORS["border"],
    tickfont      = dict(color=COLORS["muted"]),
)

LEGEND_STYLE = dict(
    bgcolor     = COLORS["surface2"],
    bordercolor = COLORS["border"],
    borderwidth = 1,
)
#-------------------------------------------------------------------------------------
# Helper: Market Detector
#-------------------------------------------------------------------------------------
"""
    Detect whether tickers belong to India, US, Europe or are mixed.
    Returns: tuple : (market_code, alert_type, alert_message)
"""
def detect_market(tickers: list) -> tuple:    
    ns_bo  = [t for t in tickers if t.endswith((".NS", ".BO"))]
    us     = [t for t in tickers if "." not in t]
    eu     = [t for t in tickers if any(t.endswith(s) for s in (".DE",".PA",".L",".MI",".AS"))]

    if len(ns_bo) == len(tickers):
        return "IN", "success", "🟢 All Indian tickers - using RBI Repo Rate (5.25%)"
    elif len(us) == len(tickers):
        return "US", "warning", "🟡 All US tickers - using Fed Funds Rate (4.30%)"
    elif len(eu) == len(tickers):
        return "EU", "warning", "🟡 All European tickers - using ECB Rate (2.50%)"
    else:
        return "MIXED", "danger",  "🔴 Mixed markets detected - please use tickers from one market only. Results will be unreliable."
#-------------------------------------------------------------------------------------
"""Return (alert_type, message) based on number of trading days."""
def data_quality_banner(n_days: int) -> tuple:    
    if n_days < MIN_TRADING_DAYS:
        return "danger",  f"🔴 Only {n_days} trading days. Minimum 3 years ({MIN_TRADING_DAYS} days) required for reliable results."
    elif n_days < WARN_TRADING_DAYS:
        return "warning", f"🟡 {n_days} trading days (< 5 years). CDaR may underestimate tail risk - consider extending the date range."
    elif n_days < 2520:
        return "success", f"🟢 {n_days} trading days. All models reliable."
    else:
        return "success", f"🟢 {n_days} trading days - excellent coverage. Full market cycle included."

#-------------------------------------------------------------------------------------
# App Initialisation
#-------------------------------------------------------------------------------------
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap",
    ],
    title="Portfolio Optimisation Dashboard",
    suppress_callback_exceptions=True,
)
server = app.server   # expose for deployment

#-------------------------------------------------------------------------------------
# Layout Components
#-------------------------------------------------------------------------------------
"""Reusable KPI metric card."""
def kpi_card(title: str, value: str, subtitle: str = "", color: str = "primary") -> dbc.Card:
    color_map = {
        "primary": COLORS["primary"],
        "green":   COLORS["green"],
        "orange":  COLORS["orange"],
        "red":     COLORS["red"],
        "purple":  COLORS["purple"],
    }
    accent = color_map.get(color, COLORS["primary"])
    return dbc.Card(
        dbc.CardBody([
            html.P(title, className="mb-1",
                   style={"color": COLORS["muted"], "fontSize": "0.75rem",
                          "textTransform": "uppercase", "letterSpacing": "0.08em"}),
            html.H4(value, className="mb-0",
                    style={"color": accent, "fontWeight": "700", "fontSize": "1.4rem"}),
            html.P(subtitle, className="mb-0",
                   style={"color": COLORS["muted"], "fontSize": "0.7rem"}) if subtitle else html.Div(),
        ]),
        style={
            "background":   COLORS["surface"],
            "border":       f"1px solid {COLORS['border']}",
            "borderRadius": "8px",
            "borderTop":    f"3px solid {accent}",
        },
        className="h-100",
    )
#-------------------------------------------------------------------------------------
"""Left sidebar with all user controls."""
def sidebar() -> html.Div:
    return html.Div([

        # Logo / Title
        html.Div([
            html.Div("📈", style={"fontSize": "2rem"}),
            html.H5("Portfolio Optimiser", className="mb-0",
                    style={"fontWeight": "700", "color": COLORS["text"]}),
            html.P("Markowitz · CDaR · SHAP XAI",
                   style={"color": COLORS["muted"], "fontSize": "0.72rem", "margin": "0"}),
        ], className="d-flex align-items-center gap-3 mb-4",
           style={"paddingBottom": "1rem", "borderBottom": f"1px solid {COLORS['border']}"}),

        # Market Selector
        html.Label("Market", style={"color": COLORS["muted"], "fontSize": "0.75rem",
                                    "textTransform": "uppercase", "letterSpacing": "0.06em"}),
        dcc.Dropdown(
            id="market-selector",
            options=[{"label": v, "value": k} for k, v in MARKET_LABELS.items()
                     if k != "MIXED"],
            value="IN",
            clearable=False,
            style={"marginBottom": "1rem"},
        ),

        # Ticker Input
        html.Label("Tickers", style={"color": COLORS["muted"], "fontSize": "0.75rem",
                                     "textTransform": "uppercase", "letterSpacing": "0.06em"}),
        dcc.Textarea(
            id="ticker-input",
            value=DEFAULT_TICKERS,
            placeholder="e.g. HDFCBANK.NS, ICICIBANK.NS, SBIN.NS",
            style={
                "width":        "100%",
                "height":       "80px",
                "background":   COLORS["surface2"],
                "border":       f"1px solid {COLORS['border']}",
                "borderRadius": "6px",
                "color":        COLORS["text"],
                "fontSize":     "0.82rem",
                "padding":      "8px",
                "resize":       "vertical",
                "marginBottom": "0.5rem",
            },
        ),
        html.P("Comma-separated. NSE: add .NS  |  BSE: add .BO",
               style={"color": COLORS["muted"], "fontSize": "0.68rem", "marginBottom": "1rem"}),

        # Date Range
        html.Label("Date Range", style={"color": COLORS["muted"], "fontSize": "0.75rem",
                                        "textTransform": "uppercase", "letterSpacing": "0.06em"}),
        dbc.Row([
            dbc.Col(dcc.Input(id="start-date", type="text", value=DEFAULT_START,
                              placeholder="YYYY-MM-DD",
                              style={"width": "100%", "background": COLORS["surface2"],
                                     "border": f"1px solid {COLORS['border']}",
                                     "borderRadius": "6px", "color": COLORS["text"],
                                     "fontSize": "0.8rem", "padding": "6px 8px"}), width=6),
            dbc.Col(dcc.Input(id="end-date", type="text", value=DEFAULT_END,
                              placeholder="YYYY-MM-DD",
                              style={"width": "100%", "background": COLORS["surface2"],
                                     "border": f"1px solid {COLORS['border']}",
                                     "borderRadius": "6px", "color": COLORS["text"],
                                     "fontSize": "0.8rem", "padding": "6px 8px"}), width=6),
        ], className="mb-3"),

        # CDaR Beta
        html.Label("CDaR Confidence (β)",
                   style={"color": COLORS["muted"], "fontSize": "0.75rem",
                          "textTransform": "uppercase", "letterSpacing": "0.06em"}),
        dcc.Slider(
            id="beta-slider",
            min=0.80, max=0.99, step=0.01, value=0.95,
            marks={0.80: "0.80", 0.90: "0.90", 0.95: "0.95", 0.99: "0.99"},
            tooltip={"placement": "bottom", "always_visible": False},
        ),
        html.P("0.95 = average of worst 5% drawdowns",
               style={"color": COLORS["muted"], "fontSize": "0.68rem",
                      "marginTop": "0.25rem", "marginBottom": "1rem"}),

        # Weight Bounds
        html.Label("Weight Bounds per Asset",
                   style={"color": COLORS["muted"], "fontSize": "0.75rem",
                          "textTransform": "uppercase", "letterSpacing": "0.06em"}),
        dbc.Row([
            dbc.Col([
                html.P("Min %", style={"color": COLORS["muted"],
                                       "fontSize": "0.68rem", "marginBottom": "2px"}),
                dcc.Input(id="min-weight", type="number", value=0, min=0, max=50, step=1,
                          style={"width": "100%", "background": COLORS["surface2"],
                                 "border": f"1px solid {COLORS['border']}",
                                 "borderRadius": "6px", "color": COLORS["text"],
                                 "fontSize": "0.8rem", "padding": "6px 8px"}),
            ], width=6),
            dbc.Col([
                html.P("Max %", style={"color": COLORS["muted"],
                                       "fontSize": "0.68rem", "marginBottom": "2px"}),
                dcc.Input(id="max-weight", type="number", value=100, min=10, max=100, step=5,
                          style={"width": "100%", "background": COLORS["surface2"],
                                 "border": f"1px solid {COLORS['border']}",
                                 "borderRadius": "6px", "color": COLORS["text"],
                                 "fontSize": "0.8rem", "padding": "6px 8px"}),
            ], width=6),
        ], className="mb-3"),

        # Run Button
        dbc.Button(
            "▶  Run Analysis",
            id="run-btn",
            color="primary",
            className="w-100 mb-3",
            style={"fontWeight": "600", "letterSpacing": "0.04em",
                   "borderRadius": "6px", "padding": "10px"},
        ),

        # Status Alert
        html.Div(id="status-alert"),

        # Waterfall Selector
        html.Div([
            html.Div(style={"borderTop": f"1px solid {COLORS['border']}",
                            "margin": "1rem 0"}),
            html.Label("Waterfall - Explain Ticker",
                       style={"color": COLORS["muted"], "fontSize": "0.75rem",
                              "textTransform": "uppercase", "letterSpacing": "0.06em"}),
            dcc.Dropdown(
                id="waterfall-ticker",
                options=[],
                placeholder="Select ticker after running...",
                style={"marginBottom": "0.5rem"},
            ),
        ]),

    ], style={
        "width":      "280px",
        "minWidth":   "280px",
        "background": COLORS["surface"],
        "borderRight": f"1px solid {COLORS['border']}",
        "padding":    "1.5rem 1rem",
        "height":     "100vh",
        "overflowY":  "auto",
        "position":   "fixed",
        "top":        "0",
        "left":       "0",
    })
#-------------------------------------------------------------------------------------
"""Right-side main content area."""
def main_content() -> html.Div:
    return html.Div([

        # Header
        html.Div([
            html.H4("Portfolio Optimisation · Drawdown · SHAP XAI",
                    style={"color": COLORS["text"], "fontWeight": "700",
                           "margin": "0", "fontSize": "1.1rem"}),
            html.P("Markowitz Mean-Variance  ·  Conditional Drawdown-at-Risk  ·  SHAP Attribution",
                   style={"color": COLORS["muted"], "margin": "0", "fontSize": "0.78rem"}),
        ], style={
            "padding":       "1rem 1.5rem",
            "borderBottom":  f"1px solid {COLORS['border']}",
            "background":    COLORS["surface"],
            "position":      "sticky",
            "top":           "0",
            "zIndex":        "100",
        }),

        # Loading Wrapper
        dcc.Loading(
            id="loading-main",
            type="circle",
            color=COLORS["primary"],
            children=html.Div([

                # Data Quality / Market Banner
                html.Div(id="banner-area", style={"padding": "0.75rem 1.5rem 0"}),

                # KPI Row
                html.Div(id="kpi-row", style={"padding": "0.75rem 1.5rem"}),

                # Row 1: Efficient Frontier + CDaR Frontier
                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("Markowitz Efficient Frontier",
                                       style={"background": COLORS["surface2"],
                                              "color": COLORS["text"],
                                              "borderBottom": f"1px solid {COLORS['border']}",
                                              "fontWeight": "600", "fontSize": "0.85rem"}),
                        dbc.CardBody(dcc.Graph(id="frontier-chart",
                                               config={"displayModeBar": False},
                                               style={"height": "360px"})),
                    ], style={"background": COLORS["surface"],
                              "border": f"1px solid {COLORS['border']}",
                              "borderRadius": "8px"}), md=6),

                    dbc.Col(dbc.Card([
                        dbc.CardHeader("CDaR Efficient Frontier",
                                       style={"background": COLORS["surface2"],
                                              "color": COLORS["text"],
                                              "borderBottom": f"1px solid {COLORS['border']}",
                                              "fontWeight": "600", "fontSize": "0.85rem"}),
                        dbc.CardBody(dcc.Graph(id="cdar-frontier-chart",
                                               config={"displayModeBar": False},
                                               style={"height": "360px"})),
                    ], style={"background": COLORS["surface"],
                              "border": f"1px solid {COLORS['border']}",
                              "borderRadius": "8px"}), md=6),
                ], className="g-3 px-3 mb-3"),

                # Row 2: Portfolio Weights + Drawdown Curve
                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("Portfolio Weights - All Strategies",
                                       style={"background": COLORS["surface2"],
                                              "color": COLORS["text"],
                                              "borderBottom": f"1px solid {COLORS['border']}",
                                              "fontWeight": "600", "fontSize": "0.85rem"}),
                        dbc.CardBody(dcc.Graph(id="weights-chart",
                                               config={"displayModeBar": False},
                                               style={"height": "340px"})),
                    ], style={"background": COLORS["surface"],
                              "border": f"1px solid {COLORS['border']}",
                              "borderRadius": "8px"}), md=6),

                    dbc.Col(dbc.Card([
                        dbc.CardHeader("Portfolio Drawdown - Underwater Curve",
                                       style={"background": COLORS["surface2"],
                                              "color": COLORS["text"],
                                              "borderBottom": f"1px solid {COLORS['border']}",
                                              "fontWeight": "600", "fontSize": "0.85rem"}),
                        dbc.CardBody(dcc.Graph(id="drawdown-chart",
                                               config={"displayModeBar": False},
                                               style={"height": "340px"})),
                    ], style={"background": COLORS["surface"],
                              "border": f"1px solid {COLORS['border']}",
                              "borderRadius": "8px"}), md=6),
                ], className="g-3 px-3 mb-3"),

                # Row 3: SHAP Importance + SHAP Direction
                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("SHAP Global Feature Importance (CDaR Model)",
                                       style={"background": COLORS["surface2"],
                                              "color": COLORS["text"],
                                              "borderBottom": f"1px solid {COLORS['border']}",
                                              "fontWeight": "600", "fontSize": "0.85rem"}),
                        dbc.CardBody(dcc.Graph(id="shap-importance-chart",
                                               config={"displayModeBar": False},
                                               style={"height": "340px"})),
                    ], style={"background": COLORS["surface"],
                              "border": f"1px solid {COLORS['border']}",
                              "borderRadius": "8px"}), md=6),

                    dbc.Col(dbc.Card([
                        dbc.CardHeader("SHAP Signed Direction - Feature → Weight Impact",
                                       style={"background": COLORS["surface2"],
                                              "color": COLORS["text"],
                                              "borderBottom": f"1px solid {COLORS['border']}",
                                              "fontWeight": "600", "fontSize": "0.85rem"}),
                        dbc.CardBody(dcc.Graph(id="shap-direction-chart",
                                               config={"displayModeBar": False},
                                               style={"height": "340px"})),
                    ], style={"background": COLORS["surface"],
                              "border": f"1px solid {COLORS['border']}",
                              "borderRadius": "8px"}), md=6),
                ], className="g-3 px-3 mb-3"),

                # Row 4: SHAP Waterfall
                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("SHAP Waterfall - Single Ticker Explanation",
                                       style={"background": COLORS["surface2"],
                                              "color": COLORS["text"],
                                              "borderBottom": f"1px solid {COLORS['border']}",
                                              "fontWeight": "600", "fontSize": "0.85rem"}),
                        dbc.CardBody(dcc.Graph(id="waterfall-chart",
                                               config={"displayModeBar": False},
                                               style={"height": "340px"})),
                    ], style={"background": COLORS["surface"],
                              "border": f"1px solid {COLORS['border']}",
                              "borderRadius": "8px"}), md=12),
                ], className="g-3 px-3 mb-3"),

                # Row 5: Tables
                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("Strategy Comparison",
                                       style={"background": COLORS["surface2"],
                                              "color": COLORS["text"],
                                              "borderBottom": f"1px solid {COLORS['border']}",
                                              "fontWeight": "600", "fontSize": "0.85rem"}),
                        dbc.CardBody(html.Div(id="strategy-table")),
                    ], style={"background": COLORS["surface"],
                              "border": f"1px solid {COLORS['border']}",
                              "borderRadius": "8px"}), md=6),

                    dbc.Col(dbc.Card([
                        dbc.CardHeader("Per-Asset Drawdown Table",
                                       style={"background": COLORS["surface2"],
                                              "color": COLORS["text"],
                                              "borderBottom": f"1px solid {COLORS['border']}",
                                              "fontWeight": "600", "fontSize": "0.85rem"}),
                        dbc.CardBody(html.Div(id="asset-table")),
                    ], style={"background": COLORS["surface"],
                              "border": f"1px solid {COLORS['border']}",
                              "borderRadius": "8px"}), md=6),
                ], className="g-3 px-3 mb-4"),

            ]),
        ),

    ], style={
        "marginLeft": "280px",
        "background": COLORS["bg"],
        "minHeight":  "100vh",
    })

#-------------------------------------------------------------------------------------
# App Layout
#-------------------------------------------------------------------------------------
app.layout = html.Div([
    dcc.Store(id="results-store"),   # stores serialisable results between callbacks
    sidebar(),
    main_content(),
], style={"fontFamily": "Inter, sans-serif", "background": COLORS["bg"]})

#-------------------------------------------------------------------------------------
# Chart Builders
#-------------------------------------------------------------------------------------

def build_frontier_chart(mk_results: dict) -> go.Figure:
    frontier = mk_results["frontier"]
    ms       = mk_results["max_sharpe"]
    mv       = mk_results["min_vol"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frontier["annual_vol"] * 100, y=frontier["annual_return"] * 100,
        mode="lines", name="Efficient Frontier",
        line=dict(color=COLORS["primary"], width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=[ms["annual_vol"] * 100], y=[ms["annual_return"] * 100],
        mode="markers+text", name=f"Max Sharpe (SR={ms['sharpe_ratio']:.2f})",
        marker=dict(color=COLORS["max_sharpe"], size=12, symbol="star"),
        text=["Max Sharpe"], textposition="top right",
        textfont=dict(color=COLORS["max_sharpe"], size=10),
    ))
    fig.add_trace(go.Scatter(
        x=[mv["annual_vol"] * 100], y=[mv["annual_return"] * 100],
        mode="markers+text", name=f"Min Vol (SR={mv['sharpe_ratio']:.2f})",
        marker=dict(color=COLORS["min_vol"], size=12, symbol="diamond"),
        text=["Min Vol"], textposition="top right",
        textfont=dict(color=COLORS["min_vol"], size=10),
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        xaxis=dict(**AXIS_STYLE, title="Annual Volatility (%)"),
        yaxis=dict(**AXIS_STYLE, title="Annual Return (%)"),
    )
    return fig
#-------------------------------------------------------------------------------------

def build_cdar_frontier_chart(dd_results: dict) -> go.Figure:
    frontier = dd_results["cdar_frontier"]
    min_cdar = dd_results["min_cdar"]
    mk_dd    = dd_results.get("markowitz_dd")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frontier["cdar"] * 100, y=frontier["annual_return"] * 100,
        mode="lines", name="CDaR Frontier",
        line=dict(color=COLORS["orange"], width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=[min_cdar["cdar"] * 100], y=[min_cdar["annual_return"] * 100],
        mode="markers+text", name=f"Min CDaR (Calmar={min_cdar['calmar_ratio']:.2f})",
        marker=dict(color=COLORS["min_cdar"], size=12, symbol="star"),
        text=["Min CDaR"], textposition="top right",
        textfont=dict(color=COLORS["min_cdar"], size=10),
    ))
    if mk_dd:
        fig.add_trace(go.Scatter(
            x=[mk_dd["cdar"] * 100], y=[min_cdar["annual_return"] * 100],
            mode="markers+text", name=f"Max Sharpe CDaR={mk_dd['cdar']:.1%}",
            marker=dict(color=COLORS["red"], size=10, symbol="x"),
            text=["Max Sharpe"], textposition="top right",
            textfont=dict(color=COLORS["red"], size=10),
        ))
    fig.update_layout(
        **CHART_LAYOUT,
        xaxis=dict(**AXIS_STYLE, title="CDaR (%)"),
        yaxis=dict(**AXIS_STYLE, title="Annual Return (%)"),
    )
    return fig
#-------------------------------------------------------------------------------------

def build_drawdown_chart(returns: pd.DataFrame, mk_results: dict, dd_results: dict) -> go.Figure:
    strategies = {
        "Max Sharpe":   (mk_results["max_sharpe"]["weights"],   COLORS["max_sharpe"]),
        "Min Vol":      (mk_results["min_vol"]["weights"],      COLORS["min_vol"]),
        "Min CDaR":     (dd_results["min_cdar"]["weights"],     COLORS["min_cdar"]),
        "Equal Weight": (mk_results["equal_weight"]["weights"], COLORS["equal_w"]),
    }

    fig = go.Figure()
    for name, (weights, color) in strategies.items():
        dd_curve = compute_portfolio_drawdown(returns, weights)
        fig.add_trace(go.Scatter(
            x=dd_curve.index, y=dd_curve.values * 100,
            mode="lines", name=name,
            line=dict(color=color, width=1.8),
            fill="tozeroy" if name == "Min CDaR" else "none",
            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:],16)},0.07)",
        ))

    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["border"])
    fig.update_layout(
        **CHART_LAYOUT,
        xaxis=dict(**AXIS_STYLE, title="Date"),
        yaxis=dict(**AXIS_STYLE, title="Drawdown (%)", ticksuffix="%"),
    )
    return fig
#-------------------------------------------------------------------------------------

def build_weights_chart(mk_results: dict, dd_results: dict) -> go.Figure:
    tickers = list(mk_results["max_sharpe"]["weights"].index)
    ms_w    = [mk_results["max_sharpe"]["weights"].get(t, 0) * 100 for t in tickers]
    mv_w    = [mk_results["min_vol"]["weights"].get(t, 0) * 100 for t in tickers]
    eq_w    = [mk_results["equal_weight"]["weights"].get(t, 0) * 100 for t in tickers]
    cdar_w  = [dd_results["min_cdar"]["weights"].get(t, 0) * 100 for t in tickers]

    fig = go.Figure()
    for weights, name, color in [
        (ms_w,   "Max Sharpe",   COLORS["max_sharpe"]),
        (mv_w,   "Min Vol",      COLORS["min_vol"]),
        (cdar_w, "Min CDaR",     COLORS["min_cdar"]),
        (eq_w,   "Equal Weight", COLORS["equal_w"]),
    ]:
        fig.add_trace(go.Bar(
            name=name, x=tickers, y=weights,
            marker_color=color, opacity=0.85,
        ))
    fig.update_layout(
        **CHART_LAYOUT,
        barmode="group",
        xaxis=dict(**AXIS_STYLE, title="Ticker"),
        yaxis=dict(**AXIS_STYLE, title="Weight (%)"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor=COLORS["surface2"], bordercolor=COLORS["border"], borderwidth=1,
        ),
    )
    return fig
#-------------------------------------------------------------------------------------

def build_shap_importance_chart(shap_results: dict) -> go.Figure:
    gi = shap_results["global_importance"]
    gi = gi[~gi["feature"].str.startswith("ticker_")].head(10)

    fig = go.Figure(go.Bar(
        x=gi["mean_abs_shap"], y=gi["feature"],
        orientation="h",
        marker=dict(
            color=gi["mean_abs_shap"],
            colorscale=[[0, COLORS["surface2"]], [1, COLORS["primary"]]],
            showscale=False,
        ),
        text=[f"{v:.6f}" for v in gi["mean_abs_shap"]],
        textposition="outside",
        textfont=dict(color=COLORS["muted"], size=10),
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        xaxis=dict(**AXIS_STYLE, title="Mean |SHAP Value|"),
        yaxis=dict(**AXIS_STYLE, autorange="reversed"),
        legend=dict(**LEGEND_STYLE),
    )
    return fig
#-------------------------------------------------------------------------------------

def build_shap_direction_chart(shap_results: dict) -> go.Figure:
    dir_table = shap_results["direction_table"]
    col_mask  = dir_table.abs().max() > 1e-6
    dir_table = dir_table.loc[:, col_mask]

    fig = go.Figure(go.Heatmap(
        z=dir_table.values,
        x=dir_table.columns.tolist(),
        y=dir_table.index.tolist(),
        colorscale=[
            [0.0, COLORS["red"]],
            [0.5, COLORS["surface2"]],
            [1.0, COLORS["green"]],
        ],
        zmid=0,
        text=np.round(dir_table.values, 5),
        texttemplate="%{text}",
        textfont=dict(size=9, color=COLORS["text"]),
        colorbar=dict(
            title=dict(
                text="SHAP",
                font=dict(color=COLORS["muted"], size=10),  # replaces titlefont
            ),
            tickfont=dict(color=COLORS["muted"], size=9),
            bgcolor=COLORS["surface"],
        ),
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        xaxis=dict(**AXIS_STYLE, title="Feature", tickangle=-30),
        yaxis=dict(**AXIS_STYLE, title="Ticker"),
        legend=dict(**LEGEND_STYLE),
    )
    return fig
#-------------------------------------------------------------------------------------

def build_waterfall_chart(shap_results: dict, ticker: str) -> go.Figure:
    wf = waterfall_data(
        shap_results["cdar_explainer"],
        shap_results["X"],
        shap_results["meta_df"],
        ticker,
        shap_results["feature_names"],
    )
    if not wf:
        return go.Figure().update_layout(**CHART_LAYOUT)

    features  = wf["features"][:8]
    shap_vals = wf["shap_values"][:8]
    feat_vals = wf["feature_values"][:8]

    labels = [f"{f}<br><span style='font-size:9px'>(val={v:.3f})</span>"
              for f, v in zip(features, feat_vals)]

    fig = go.Figure(go.Waterfall(
        orientation="h",
        measure=["relative"] * len(shap_vals) + ["total"],
        x=list(shap_vals) + [sum(shap_vals)],
        y=labels + [f"<b>{ticker} Prediction</b>"],
        connector=dict(line=dict(color=COLORS["border"], width=1)),
        increasing=dict(marker=dict(color=COLORS["green"])),
        decreasing=dict(marker=dict(color=COLORS["red"])),
        totals=dict(marker=dict(color=COLORS["primary"])),
        text=[f"{v:+.5f}" for v in shap_vals] + [f"{sum(shap_vals):+.5f}"],
        textposition="outside",
        textfont=dict(color=COLORS["text"], size=10),
    ))

    fig.update_layout(
        **CHART_LAYOUT,
        xaxis=dict(**AXIS_STYLE, title="SHAP Value (impact on predicted weight)"),
        yaxis=dict(**AXIS_STYLE, autorange="reversed"),
        legend=dict(**LEGEND_STYLE),
        title=dict(
            text=f"SHAP Waterfall - {ticker}  |  Base: {wf['base_value']:.4f}  |  Date: {str(wf['date'])[:10]}",
            font=dict(size=11, color=COLORS["muted"]),
            x=0,
        ),
    )
    return fig
#-------------------------------------------------------------------------------------
"""Reusable styled DataTable."""
def build_dash_table(df: pd.DataFrame, table_id: str) -> dash_table.DataTable:
    
    df = df.reset_index()
    return dash_table.DataTable(
        id=table_id,
        columns=[{"name": c, "id": c} for c in df.columns],
        data=df.to_dict("records"),
        style_table={"overflowX": "auto"},
        style_cell={
            "background":  COLORS["surface"],
            "color":       COLORS["text"],
            "border":      f"1px solid {COLORS['border']}",
            "padding":     "6px 10px",
            "fontSize":    "0.78rem",
            "fontFamily":  "Inter, sans-serif",
            "textAlign":   "center",
        },
        style_header={
            "background":  COLORS["surface2"],
            "color":       COLORS["muted"],
            "fontWeight":  "600",
            "fontSize":    "0.72rem",
            "textTransform": "uppercase",
            "letterSpacing": "0.05em",
            "border":      f"1px solid {COLORS['border']}",
        },
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "background": COLORS["surface2"]},
        ],
        page_action="none",
    )

#-------------------------------------------------------------------------------------
# Main Callback
#-------------------------------------------------------------------------------------
@app.callback(
    Output("results-store",         "data"),
    Output("banner-area",           "children"),
    Output("kpi-row",               "children"),
    Output("frontier-chart",        "figure"),
    Output("cdar-frontier-chart",   "figure"),
    Output("weights-chart",         "figure"),
    Output("drawdown-chart",        "figure"),
    Output("shap-importance-chart", "figure"),
    Output("shap-direction-chart",  "figure"),
    Output("strategy-table",        "children"),
    Output("asset-table",           "children"),
    Output("waterfall-ticker",      "options"),
    Output("waterfall-ticker",      "value"),
    Output("status-alert",          "children"),
    Input("run-btn",                "n_clicks"),
    State("ticker-input",           "value"),
    State("start-date",             "value"),
    State("end-date",               "value"),
    State("market-selector",        "value"),
    State("beta-slider",            "value"),
    State("min-weight",             "value"),
    State("max-weight",             "value"),
    prevent_initial_call=True,
)
#-------------------------------------------------------------------------------------

def run_analysis(n_clicks, ticker_str, start_date, end_date,
                 market, beta, min_w, max_w):

    empty_fig = go.Figure().update_layout(**CHART_LAYOUT)

    # Parse tickers
    tickers = [t.strip().upper() for t in ticker_str.replace("\n", ",").split(",")
               if t.strip()]
    if len(tickers) < 2:
        alert = dbc.Alert("⚠️ Please enter at least 2 tickers.", color="warning",
                          dismissable=True)
        return (None, [], [], empty_fig, empty_fig, empty_fig, empty_fig,
                empty_fig, empty_fig, [], [], [], None, alert)

    # Market detection
    mkt_code, mkt_type, mkt_msg = detect_market(tickers)
    rfr = RISK_FREE_RATES.get(mkt_code, RISK_FREE_RATES["IN"])

    if mkt_code == "MIXED":
        alert   = dbc.Alert(mkt_msg, color="danger", dismissable=True)
        banners = [dbc.Alert(mkt_msg, color="danger", className="mb-0")]
        return (None, banners, [], empty_fig, empty_fig, empty_fig, empty_fig,
                empty_fig, empty_fig, [], [], [], None, alert)

    # Weight bounds
    weight_bounds = (
        float(min_w or 0) / 100,
        float(max_w or 100) / 100,
    )

    try:
        # 1. Fetch data
        prices, returns = get_data(
            tickers=tickers,
            start=start_date,
            end=end_date,
            force_refresh=True,
        )

        n_days           = len(prices)
        dq_type, dq_msg  = data_quality_banner(n_days)

        banners = [
            dbc.Alert(mkt_msg, color=mkt_type,   dismissable=True, className="mb-2"),
            dbc.Alert(dq_msg,  color=dq_type,    dismissable=True, className="mb-0"),
        ]

        if n_days < MIN_TRADING_DAYS:
            return (None, banners, [], empty_fig, empty_fig, empty_fig, empty_fig,
                    empty_fig, empty_fig, [], [], [], None,
                    dbc.Alert(dq_msg, color="danger", dismissable=True))

        # 2. Markowitz
        mk = run_markowitz(
            prices,
            risk_free_rate=rfr,
            weight_bounds=weight_bounds,
            verbose=False,
        )

        # 3. Drawdown
        dd = run_drawdown(
            prices, returns,
            markowitz_weights=mk["max_sharpe"]["weights"].to_dict(),
            beta=beta,
            weight_bounds=weight_bounds,
            risk_free_rate=rfr,
            verbose=False,
        )

        # 4. SHAP
        shap_res = run_shap(prices, returns, mk, dd, verbose=False)

        # 5. KPI Cards
        ms  = mk["max_sharpe"]
        mv  = mk["min_vol"]
        mc  = dd["min_cdar"]

        kpi_row = dbc.Row([
            dbc.Col(kpi_card("Max Sharpe Return",
                             f"{ms['annual_return']:.1%}",
                             f"Vol: {ms['annual_vol']:.1%}  |  SR: {ms['sharpe_ratio']:.2f}",
                             "primary"), md=3),
            dbc.Col(kpi_card("Min Vol Return",
                             f"{mv['annual_return']:.1%}",
                             f"Vol: {mv['annual_vol']:.1%}  |  SR: {mv['sharpe_ratio']:.2f}",
                             "green"), md=3),
            dbc.Col(kpi_card("Min CDaR Return",
                             f"{mc['annual_return']:.1%}",
                             f"CDaR: {mc['cdar']:.1%}  |  Calmar: {mc['calmar_ratio']:.2f}",
                             "orange"), md=3),
            dbc.Col(kpi_card("Max Sharpe MDD",
                             f"{dd['markowitz_dd']['max_drawdown']:.1%}",
                             f"vs CDaR MDD: {mc['max_drawdown']:.1%}",
                             "red"), md=3),
        ], className="g-3")

        # 6. Build charts
        fig_frontier = build_frontier_chart(mk)
        fig_cdar     = build_cdar_frontier_chart(dd)
        fig_weights  = build_weights_chart(mk, dd)
        fig_dd       = build_drawdown_chart(returns, mk, dd)
        fig_shap_imp = build_shap_importance_chart(shap_res)
        fig_shap_dir = build_shap_direction_chart(shap_res)

        # 7. Tables
        summary_df   = mk["summary"].copy()
        # Add CDaR row
        cdar_row = pd.DataFrame([{
            "Ann. Return":     f"{mc['annual_return']:.2%}",
            "Ann. Volatility": f"{mc['annual_vol']:.2%}",
            "Sharpe Ratio":    f"{ms['sharpe_ratio']:.3f}",
            "Top Holding":     mc["weights"].idxmax(),
            "Top Weight":      f"{mc['weights'].max():.1%}",
        }], index=["Min CDaR"])
        summary_df = pd.concat([summary_df, cdar_row])

        tbl_strategy = build_dash_table(summary_df, "tbl-strategy")
        tbl_asset    = build_dash_table(dd["asset_table"], "tbl-asset")

        # 8. Waterfall dropdown
        wf_options = [{"label": t, "value": t} for t in tickers]
        wf_default = tickers[0]

        status_alert = dbc.Alert(
            f"✅ Analysis complete - {n_days} trading days, {len(tickers)} assets.",
            color="success", dismissable=True,
        )

        return (
            {"tickers": tickers},   # store
            banners,
            kpi_row,
            fig_frontier,
            fig_cdar,
            fig_weights,
            fig_dd,
            fig_shap_imp,
            fig_shap_dir,
            tbl_strategy,
            tbl_asset,
            wf_options,
            wf_default,
            status_alert,
        )

    except Exception as e:
        log.exception("Analysis failed")
        alert = dbc.Alert(f"❌ Error: {str(e)}", color="danger", dismissable=True)
        return (None, [], [], empty_fig, empty_fig, empty_fig, empty_fig,
                empty_fig, empty_fig, [], [], [], None, alert)

#-------------------------------------------------------------------------------------
# Waterfall Callback (separate - lightweight)
#-------------------------------------------------------------------------------------
@app.callback(
    Output("waterfall-chart", "figure"),
    Input("waterfall-ticker", "value"),
    State("ticker-input",     "value"),
    State("start-date",       "value"),
    State("end-date",         "value"),
    State("market-selector",  "value"),
    State("beta-slider",      "value"),
    State("min-weight",       "value"),
    State("max-weight",       "value"),
    prevent_initial_call=True,
)
#-------------------------------------------------------------------------------------

def update_waterfall(ticker, ticker_str, start_date, end_date,
                     market, beta, min_w, max_w):
    if not ticker:
        return go.Figure().update_layout(**CHART_LAYOUT)

    try:
        tickers = [t.strip().upper() for t in ticker_str.replace("\n", ",").split(",")
                   if t.strip()]
        mkt_code, _, _ = detect_market(tickers)
        rfr  = RISK_FREE_RATES.get(mkt_code, RISK_FREE_RATES["IN"])
        wb   = (float(min_w or 0) / 100, float(max_w or 100) / 100)

        prices, returns = get_data(tickers=tickers, start=start_date, end=end_date)
        mk   = run_markowitz(prices, risk_free_rate=rfr, weight_bounds=wb, verbose=False)
        dd   = run_drawdown(prices, returns,
                            markowitz_weights=mk["max_sharpe"]["weights"].to_dict(),
                            beta=beta, weight_bounds=wb,
                            risk_free_rate=rfr, verbose=False)
        shap_res = run_shap(prices, returns, mk, dd, verbose=False)

        return build_waterfall_chart(shap_res, ticker)

    except Exception as e:
        log.exception("Waterfall update failed")
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", showarrow=False,
                           font=dict(color=COLORS["red"]))
        return fig.update_layout(**CHART_LAYOUT)

#-------------------------------------------------------------------------------------
# Entry Point 
#-------------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=8050)

#-------------------------------------------------------------------------------------