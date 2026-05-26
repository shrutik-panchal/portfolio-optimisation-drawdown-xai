#-------------------------------------------------------------------------------------
"""
app.py
------
Dash dashboard for the Portfolio Optimisation + Drawdown + SHAP XAI project.
"""
#-------------------------------------------------------------------------------------
# Importing required libraries and modules
#-------------------------------------------------------------------------------------
import sys, os, logging, warnings
import numpy as np
import pandas as pd

import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetch_data import get_data
from models.markowitz import run_markowitz
from models.drawdown import run_drawdown, compute_portfolio_drawdown
from models.shap_attribution import run_shap, waterfall_data
#-------------------------------------------------------------------------------------
# Logging setup
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Constants
#-------------------------------------------------------------------------------------
RISK_FREE_RATES = {"IN": 0.0525, "US": 0.0430, "EU": 0.0250}
MARKET_LABELS   = {
    "IN": "IN India (NSE/BSE) — RFR 5.25%",
    "US": "US United States   — RFR 4.30%",
    "EU": "EU Europe          — RFR 2.50%",
}
DEFAULT_TICKERS = "AXISBANK.NS, HDFCBANK.NS, ICICIBANK.NS, KOTAKBANK.NS, PNB.NS, SBIN.NS"
DEFAULT_START   = "2015-01-01"
DEFAULT_END     = pd.Timestamp.today().strftime("%Y-%m-%d")
MIN_DAYS, WARN_DAYS = 756, 1260
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Design tokens
#-------------------------------------------------------------------------------------
C = {
    "bg":      "#0d0f18", "surface":  "#13161f", "surface2": "#1a1d2b",
    "border":  "#252836", "text":     "#dde1f0", "muted":    "#6b7185",
    "faint":   "#3a3d52", "primary":  "#4f98a3", "green":    "#6daa45",
    "orange":  "#fdab43", "red":      "#dd6974", "purple":   "#a86fdf",
    "gold":    "#e8af34",
}
_BASE = dict(paper_bgcolor=C["surface"], plot_bgcolor=C["surface"],
             font=dict(color=C["text"], family="Inter,sans-serif", size=12),
             margin=dict(l=55, r=20, t=36, b=50))
#-------------------------------------------------------------------------------------
"""Return layout dict with optional key overrides (avoids duplicate-kwarg errors)."""
def BASE(**overrides):
    
    d = dict(_BASE)
    d.update(overrides)
    return d
AX  = dict(gridcolor=C["border"], zerolinecolor=C["border"],
           tickfont=dict(color=C["muted"], size=11))
LEG = dict(bgcolor=C["surface2"], bordercolor=C["border"],
           borderwidth=1, font=dict(size=11))

EMPTY_FIG = go.Figure().update_layout(**BASE())
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# UI helpers
#-------------------------------------------------------------------------------------
# Input styles with optional overrides
def inp_style(extra=None):
    s = {"background": C["surface2"], "border": f"1px solid {C['border']}",
         "borderRadius": "6px", "color": C["text"],
         "fontSize": "0.82rem", "padding": "6px 10px"}
    if extra:
        s.update(extra)
    return s
#-------------------------------------------------------------------------------------
# Label for controls
def lbl(text):
    return html.P(text, style={"color": C["muted"], "fontSize": "0.68rem",
                                "textTransform": "uppercase",
                                "letterSpacing": "0.06em", "marginBottom": "3px"})
#-------------------------------------------------------------------------------------
# Wrapper for a control with its label
def ctrl(label_text, component):
    return html.Div([lbl(label_text), component])
#-------------------------------------------------------------------------------------
# Card wrapper with header and body
def card_wrap(header_text, body):
    return html.Div([
        html.Div(header_text, style={
            "background": C["surface2"], "color": C["muted"],
            "padding": "10px 16px", "fontSize": "0.75rem", "fontWeight": "600",
            "textTransform": "uppercase", "letterSpacing": "0.07em",
            "borderBottom": f"1px solid {C['border']}",
        }),
        html.Div(body),
    ], style={"background": C["surface"], "border": f"1px solid {C['border']}",
              "borderRadius": "10px", "overflow": "hidden"})
#-------------------------------------------------------------------------------------
# Graph component with default styles and empty figure
def mk_graph(gid, height=380):
    return dcc.Graph(id=gid, config={"displayModeBar": False},
                     style={"height": f"{height}px"}, figure=EMPTY_FIG)
#-------------------------------------------------------------------------------------
# KPI card component with title, value, subtext, and accent color
def kpi_card(title, val_id, sub_id, accent):
    return html.Div([
        html.P(title, style={"color": C["muted"], "fontSize": "0.68rem",
                              "textTransform": "uppercase", "letterSpacing": "0.08em",
                              "marginBottom": "4px"}),
        html.H3(id=val_id, children="—", style={"color": accent, "fontWeight": "700",
                                                  "fontSize": "1.7rem", "margin": "0 0 2px"}),
        html.P(id=sub_id, children="", style={"color": C["muted"],
                                               "fontSize": "0.7rem", "margin": "0"}),
    ], style={"background": C["surface"], "border": f"1px solid {C['border']}",
              "borderRadius": "10px", "borderTop": f"3px solid {accent}",
              "padding": "14px 18px", "flex": "1", "minWidth": "180px"})
#-------------------------------------------------------------------------------------
# DataTable builder with consistent styling
def dtable(df, tid):
    df2 = df.reset_index()
    return dash_table.DataTable(
        id=tid,
        columns=[{"name": c, "id": c} for c in df2.columns],
        data=df2.to_dict("records"),
        style_table={"overflowX": "auto"},
        style_cell={"background": C["surface"], "color": C["text"],
                    "border": f"1px solid {C['border']}", "padding": "7px 12px",
                    "fontSize": "0.78rem", "fontFamily": "Inter,sans-serif",
                    "textAlign": "center"},
        style_header={"background": C["surface2"], "color": C["muted"],
                      "fontWeight": "600", "fontSize": "0.7rem",
                      "textTransform": "uppercase", "letterSpacing": "0.05em",
                      "border": f"1px solid {C['border']}"},
        style_data_conditional=[{"if": {"row_index": "odd"},
                                  "background": C["surface2"]}],
        page_action="none",
    )
#-------------------------------------------------------------------------------------
def pct(v): return f"{v:.1%}"
def f2(v):  return f"{v:.2f}"
def f3(v):  return f"{v:.3f}"
#-------------------------------------------------------------------------------------
# Detect market based on ticker suffixes, return market code, banner type, and message
def detect_market(tickers):
    ns_bo = [t for t in tickers if t.endswith((".NS", ".BO"))]
    us    = [t for t in tickers if "." not in t]
    eu    = [t for t in tickers if any(t.endswith(s) for s in (".DE",".PA",".L",".MI",".AS"))]
    if len(ns_bo) == len(tickers): return "IN", "success", "All Indian tickers — RBI Repo Rate (5.25%)"
    if len(us)    == len(tickers): return "US", "warning", "All US tickers — Fed Funds Rate (4.30%)"
    if len(eu)    == len(tickers): return "EU", "warning", "All European tickers — ECB Rate (2.50%)"
    return "MIXED", "danger", "Mixed markets — use tickers from one market only"
#-------------------------------------------------------------------------------------
# Data quality banner based on number of trading days, with type and message
def dq_banner(n):
    if n < MIN_DAYS:  return "danger",  f"Only {n} days. Minimum 3 years ({MIN_DAYS}) required."
    if n < WARN_DAYS: return "warning", f"{n} trading days — CDaR may underestimate tail risk."
    if n < 2520:      return "success", f"{n} trading days — all models reliable."
    return "success", f"{n} trading days — excellent, full market cycle included."
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Tab styles
#-------------------------------------------------------------------------------------
TS = {"color": C["muted"], "background": C["surface"], "border": "none",
      "padding": "11px 24px", "fontSize": "0.82rem", "fontWeight": "500",
      "borderBottom": "2px solid transparent"}
TSS = {**TS, "color": C["text"], "fontWeight": "600",
       "borderBottom": f"2px solid {C['primary']}"}
PAD = {"padding": "20px 24px"}
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# App
#-------------------------------------------------------------------------------------
app = dash.Dash(__name__,
    external_stylesheets=[
        dbc.themes.CYBORG,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap",
    ],
    title="Portfolio Optimiser",
    suppress_callback_exceptions=True,
)
server = app.server
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Layout
#-------------------------------------------------------------------------------------
app.layout = html.Div([
    dcc.Store(id="shap-store"),
    
    # Top control bar
    html.Div([
        html.Div([
            # Logo
            html.Div([
                html.Span("P", style={"fontWeight":"900","fontSize":"1.3rem",
                                      "color":C["primary"],"fontFamily":"Inter,sans-serif"}),
                html.Div([
                    html.Span("Portfolio Optimiser",
                              style={"fontWeight":"700","fontSize":"0.92rem",
                                     "color":C["text"],"display":"block"}),
                    html.Span("Markowitz · CDaR · SHAP XAI",
                              style={"color":C["muted"],"fontSize":"0.63rem"}),
                ]),
            ], style={"display":"flex","alignItems":"center","gap":"8px","minWidth":"180px"}),

            ctrl("Market", dcc.Dropdown(id="market-sel",
                options=[{"label":v,"value":k} for k,v in MARKET_LABELS.items()],
                value="IN", clearable=False,
                style={"minWidth":"220px","fontSize":"0.8rem"})),

            ctrl("Tickers", dcc.Input(id="ticker-input", value=DEFAULT_TICKERS,
                type="text", style=inp_style({"width":"300px"}))),

            ctrl("From", dcc.Input(id="start-date", value=DEFAULT_START,
                type="text", style=inp_style({"width":"105px"}))),

            ctrl("To", dcc.Input(id="end-date", value=DEFAULT_END,
                type="text", style=inp_style({"width":"105px"}))),

            ctrl("CDaR \u03b2", html.Div(
                dcc.Slider(id="beta-slider", min=0.80, max=0.99, step=0.01, value=0.95,
                           marks={0.80:"0.80",0.90:"0.90",0.95:"0.95",0.99:"0.99"},
                           tooltip={"placement":"bottom","always_visible":False}),
                style={"width":"150px","paddingTop":"4px"})),

            ctrl("Min Wt%", dcc.Input(id="min-weight", value=0, type="number",
                min=0, max=50, style=inp_style({"width":"65px"}))),

            ctrl("Max Wt%", dcc.Input(id="max-weight", value=100, type="number",
                min=10, max=100, style=inp_style({"width":"65px"}))),

            html.Div([
                html.P("\u00a0", style={"fontSize":"0.68rem","marginBottom":"3px"}),
                dbc.Button("Run Analysis", id="run-btn", color="primary",
                           style={"fontWeight":"600","borderRadius":"6px",
                                  "padding":"7px 20px","whiteSpace":"nowrap"}),
            ]),
        ], style={"display":"flex","alignItems":"flex-end","gap":"12px",
                  "flexWrap":"wrap","padding":"12px 20px 8px"}),
        html.Div(id="top-banners",
                 style={"padding":"0 20px 6px","display":"flex",
                        "gap":"8px","flexWrap":"wrap"}),
    ], style={"background":C["surface"],"borderBottom":f"1px solid {C['border']}",
              "position":"sticky","top":"0","zIndex":"200"}),

    # Tabs - ALL panels rendered at startup
    dcc.Tabs(id="main-tabs", value="overview",
        style={"background":C["surface"],"borderBottom":f"1px solid {C['border']}"},
        colors={"border":C["border"],"primary":C["primary"],"background":C["surface"]},
        children=[

        # Tab 1: Overview
        dcc.Tab(label="Overview", value="overview", style=TS, selected_style=TSS,
            children=html.Div([
                html.Div([
                    kpi_card("Max Sharpe Return", "kpi-ms-val", "kpi-ms-sub", C["primary"]),
                    kpi_card("Min Vol Return",    "kpi-mv-val", "kpi-mv-sub", C["green"]),
                    kpi_card("Min CDaR Return",   "kpi-mc-val", "kpi-mc-sub", C["orange"]),
                    kpi_card("Max Sharpe MDD",    "kpi-dd-val", "kpi-dd-sub", C["red"]),
                ], style={"display":"flex","gap":"14px","flexWrap":"wrap","marginBottom":"20px"}),
                dbc.Row([
                    dbc.Col(card_wrap("Portfolio Weights — All Strategies",
                                      mk_graph("weights-chart", 370)), md=7),
                    dbc.Col(card_wrap("Strategy Comparison",
                                      html.Div(id="strategy-table",
                                               style={"padding":"12px"})), md=5),
                ], className="g-3"),
            ], style=PAD)),

        # Tab 2: Frontiers
        dcc.Tab(label="Frontiers", value="frontiers", style=TS, selected_style=TSS,
            children=html.Div([
                dbc.Row([
                    dbc.Col(card_wrap("Markowitz Efficient Frontier",
                                      mk_graph("frontier-chart", 440)), md=6),
                    dbc.Col(card_wrap("CDaR Efficient Frontier",
                                      mk_graph("cdar-frontier-chart", 440)), md=6),
                ], className="g-3"),
            ], style=PAD)),

        # Tab 3: Drawdown
        dcc.Tab(label="Drawdown", value="drawdown", style=TS, selected_style=TSS,
            children=html.Div([
                html.Div(card_wrap("Portfolio Drawdown — Underwater Curve",
                                   mk_graph("drawdown-chart", 400)),
                         style={"marginBottom":"20px"}),
                card_wrap("Per-Asset Drawdown",
                          html.Div(id="asset-table", style={"padding":"12px"})),
            ], style=PAD)),

        # Tab 4: SHAP
        dcc.Tab(label="SHAP", value="shap", style=TS, selected_style=TSS,
            children=html.Div([
                dbc.Row([
                    dbc.Col(card_wrap("SHAP Global Feature Importance (CDaR Model)",
                                      mk_graph("shap-importance-chart", 400)), md=5),
                    dbc.Col(card_wrap("SHAP Signed Direction — Feature to Weight Impact",
                                      mk_graph("shap-direction-chart", 400)), md=7),
                ], className="g-3"),
            ], style=PAD)),

        # Tab 5: Deep Dive
        dcc.Tab(label="Deep Dive", value="deepdive", style=TS, selected_style=TSS,
            children=html.Div([
                html.Div([
                    html.P("Select a ticker to explain its SHAP attribution:",
                           style={"color":C["muted"],"fontSize":"0.82rem",
                                  "marginBottom":"8px"}),
                    dcc.Dropdown(id="waterfall-ticker", options=[],
                                 placeholder="Run analysis first, then select a ticker...",
                                 style={"width":"260px","fontSize":"0.82rem"}),
                ], style={"marginBottom":"16px"}),
                card_wrap("SHAP Waterfall — Single Ticker Explanation",
                          mk_graph("waterfall-chart", 460)),
            ], style=PAD)),

    ]),

], style={"fontFamily":"Inter,sans-serif","background":C["bg"],"minHeight":"100vh"})

#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Chart builders
#-------------------------------------------------------------------------------------
def build_frontier(mk):
    f  = mk["frontier"]
    ms, mv = mk["max_sharpe"], mk["min_vol"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f["annual_vol"]*100, y=f["annual_return"]*100,
        mode="lines", name="Efficient Frontier",
        line=dict(color=C["primary"], width=2.5)))
    fig.add_trace(go.Scatter(x=[ms["annual_vol"]*100], y=[ms["annual_return"]*100],
        mode="markers+text", name=f"Max Sharpe (SR={f2(ms['sharpe_ratio'])})",
        marker=dict(color=C["primary"], size=13, symbol="star"),
        text=["Max Sharpe"], textposition="top right",
        textfont=dict(color=C["primary"], size=10)))
    fig.add_trace(go.Scatter(x=[mv["annual_vol"]*100], y=[mv["annual_return"]*100],
        mode="markers+text", name=f"Min Vol (SR={f2(mv['sharpe_ratio'])})",
        marker=dict(color=C["green"], size=13, symbol="diamond"),
        text=["Min Vol"], textposition="top right",
        textfont=dict(color=C["green"], size=10)))
    fig.update_layout(**BASE(), legend=LEG,
        xaxis=dict(**AX, title="Annual Volatility (%)"),
        yaxis=dict(**AX, title="Annual Return (%)"))
    return fig
#-------------------------------------------------------------------------------------
def build_cdar(dd):
    fr   = dd["cdar_frontier"]
    mc   = dd["min_cdar"]
    mk_d = dd.get("markowitz_dd")
    fig  = go.Figure()
    fig.add_trace(go.Scatter(x=fr["cdar"]*100, y=fr["annual_return"]*100,
        mode="lines", name="CDaR Frontier",
        line=dict(color=C["orange"], width=2.5)))
    fig.add_trace(go.Scatter(x=[mc["cdar"]*100], y=[mc["annual_return"]*100],
        mode="markers+text", name=f"Min CDaR (Calmar={f2(mc['calmar_ratio'])})",
        marker=dict(color=C["orange"], size=13, symbol="star"),
        text=["Min CDaR"], textposition="top right",
        textfont=dict(color=C["orange"], size=10)))
    if mk_d:
        fig.add_trace(go.Scatter(x=[mk_d["cdar"]*100], y=[mc["annual_return"]*100],
            mode="markers+text", name=f"Max Sharpe CDaR={pct(mk_d['cdar'])}",
            marker=dict(color=C["red"], size=11, symbol="x"),
            text=["Max Sharpe"], textposition="top right",
            textfont=dict(color=C["red"], size=10)))
    fig.update_layout(**BASE(), legend=LEG,
        xaxis=dict(**AX, title="CDaR (%)"),
        yaxis=dict(**AX, title="Annual Return (%)"))
    return fig
#-------------------------------------------------------------------------------------
def build_weights(mk, dd):
    tickers = list(mk["max_sharpe"]["weights"].index)
    fig = go.Figure()
    for name, w, color in [
        ("Max Sharpe",   mk["max_sharpe"]["weights"],   C["primary"]),
        ("Min Vol",      mk["min_vol"]["weights"],       C["green"]),
        ("Min CDaR",     dd["min_cdar"]["weights"],      C["orange"]),
        ("Equal Weight", mk["equal_weight"]["weights"],  C["purple"]),
    ]:
        fig.add_trace(go.Bar(name=name, x=tickers,
                             y=[w.get(t, 0)*100 for t in tickers],
                             marker_color=color, opacity=0.87))
    fig.update_layout(**BASE(), barmode="group",
        xaxis=dict(**AX, title="Ticker"),
        yaxis=dict(**AX, title="Weight (%)"),
        legend=dict(**LEG, orientation="h", yanchor="top",
                    y=-0.18, xanchor="center", x=0.5))
    return fig
#-------------------------------------------------------------------------------------
def build_drawdown(returns, mk, dd):
    fig = go.Figure()
    for name, w, color in [
        ("Max Sharpe",   mk["max_sharpe"]["weights"],   C["primary"]),
        ("Min Vol",      mk["min_vol"]["weights"],       C["green"]),
        ("Min CDaR",     dd["min_cdar"]["weights"],      C["orange"]),
        ("Equal Weight", mk["equal_weight"]["weights"],  C["purple"]),
    ]:
        curve = compute_portfolio_drawdown(returns, w)
        r,g,b = int(color[1:3],16), int(color[3:5],16), int(color[5:],16)
        fig.add_trace(go.Scatter(
            x=curve.index, y=curve.values*100, mode="lines", name=name,
            line=dict(color=color, width=1.8),
            fill="tozeroy" if name == "Min CDaR" else "none",
            fillcolor=f"rgba({r},{g},{b},0.06)"))
    fig.add_hline(y=0, line_dash="dot", line_color=C["faint"])
    fig.update_layout(**BASE(), legend=LEG,
        xaxis=dict(**AX, title="Date"),
        yaxis=dict(**AX, title="Drawdown (%)", ticksuffix="%"))
    return fig
#-------------------------------------------------------------------------------------
def build_shap_imp(sr):
    gi = sr["global_importance"]
    gi = gi[~gi["feature"].str.startswith("ticker_")].head(10)
    fig = go.Figure(go.Bar(
        x=gi["mean_abs_shap"], y=gi["feature"], orientation="h",
        marker=dict(color=gi["mean_abs_shap"],
                    colorscale=[[0, C["surface2"]], [1, C["primary"]]],
                    showscale=False),
        text=[f"{v:.5f}" for v in gi["mean_abs_shap"]],
        textposition="outside", textfont=dict(color=C["muted"], size=10)))
    fig.update_layout(**BASE(margin=dict(l=130, r=60, t=36, b=50)), legend=LEG,
        xaxis=dict(**AX, title="Mean |SHAP Value|"),
        yaxis=dict(**AX, autorange="reversed"))
    return fig
#-------------------------------------------------------------------------------------
def build_shap_dir(sr):
    dt   = sr["direction_table"]
    dt   = dt.loc[:, dt.abs().max() > 1e-6]
    z    = np.round(dt.values, 4)
    zmax = max(float(np.abs(z).max()), 1e-8)
    fig  = go.Figure(go.Heatmap(
        z=z, x=dt.columns.tolist(), y=dt.index.tolist(),
        colorscale=[[0.0, C["red"]], [0.5, C["surface2"]], [1.0, C["green"]]],
        zmid=0, zmin=-zmax, zmax=zmax,
        text=[[f"{v:.4f}" if abs(v) >= 0.0001 else "" for v in row] for row in z.tolist()],
        texttemplate="%{text}",
        textfont=dict(size=10, color=C["text"]),
        colorbar=dict(
            title=dict(text="SHAP", font=dict(color=C["muted"], size=10)),
            tickfont=dict(color=C["muted"], size=9),
            bgcolor=C["surface"]),
    ))
    fig.update_layout(**BASE(margin=dict(l=110, r=20, t=36, b=110)),
        legend=LEG,
        xaxis={**AX, "title": "Feature", "tickangle": -35,
               "tickfont": dict(color=C["muted"], size=10)},
        yaxis={**AX, "title": "Ticker",
               "tickfont": dict(color=C["muted"], size=11)})
    return fig
#-------------------------------------------------------------------------------------
def build_waterfall(sr, ticker):
    wf = waterfall_data(sr["cdar_explainer"], sr["X"],
                        sr["meta_df"], ticker, sr["feature_names"])
    if not wf:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False,
                           font=dict(color=C["muted"], size=14))
        return fig.update_layout(**BASE())
    feats  = wf["features"][:10]
    svals  = wf["shap_values"][:10]
    fvals  = wf["feature_values"][:10]
    labels = [f"{f}  (val={v:.3f})" for f, v in zip(feats, fvals)]
    fig = go.Figure(go.Waterfall(
        orientation="h",
        measure=["relative"]*len(svals) + ["total"],
        x=list(svals) + [sum(svals)],
        y=labels + [f"{ticker} Prediction"],
        connector=dict(line=dict(color=C["border"], width=1)),
        increasing=dict(marker=dict(color=C["green"])),
        decreasing=dict(marker=dict(color=C["red"])),
        totals=dict(marker=dict(color=C["primary"])),
        text=[f"{v:+.5f}" for v in svals] + [f"{sum(svals):+.5f}"],
        textposition="outside", textfont=dict(color=C["text"], size=10),
    ))
    fig.update_layout(**BASE(margin=dict(l=210, r=40, t=50, b=50)),
        legend=LEG,
        title=dict(
            text=f"SHAP Waterfall — {ticker}  |  Base: {wf['base_value']:.4f}  |  Date: {str(wf['date'])[:10]}",
            font=dict(size=11, color=C["muted"]), x=0),
        xaxis=dict(**AX, title="SHAP Value (impact on predicted weight)"),
        yaxis=dict(**AX, autorange="reversed"))
    return fig

#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Main run callback
#-------------------------------------------------------------------------------------
NONE8 = [None] * 8   # 4 KPI pairs

@app.callback(
    Output("top-banners",           "children"),
    Output("kpi-ms-val",            "children"),
    Output("kpi-ms-sub",            "children"),
    Output("kpi-mv-val",            "children"),
    Output("kpi-mv-sub",            "children"),
    Output("kpi-mc-val",            "children"),
    Output("kpi-mc-sub",            "children"),
    Output("kpi-dd-val",            "children"),
    Output("kpi-dd-sub",            "children"),
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
    Output("shap-store",            "data"),
    Input("run-btn",                "n_clicks"),
    State("ticker-input",           "value"),
    State("start-date",             "value"),
    State("end-date",               "value"),
    State("market-sel",             "value"),
    State("beta-slider",            "value"),
    State("min-weight",             "value"),
    State("max-weight",             "value"),
    prevent_initial_call=True,
)
#-------------------------------------------------------------------------------------
def run_analysis(_, ticker_str, start, end, market, beta, min_w, max_w):
    def fail(msg, color="danger"):
        badge_colors = {"warning": C["gold"], "danger": C["red"], "success": C["green"]}
        badge_col = badge_colors.get(color, C["muted"])
        banner = html.Span(msg, style={
            "background": badge_col + "22", "border": f"1px solid {badge_col}55",
            "borderRadius": "20px", "color": badge_col,
            "padding": "3px 12px", "fontSize": "0.75rem", "fontWeight": "500",
        })
        return [banner] + NONE8 + [EMPTY_FIG]*6 + [[], [], None, None]

    tickers = [t.strip().upper() for t in
               ticker_str.replace("\n", ",").split(",") if t.strip()]
    if len(tickers) < 2:
        return fail("Enter at least 2 tickers.", "warning")

    mkt, mtype, mmsg = detect_market(tickers)
    rfr = RISK_FREE_RATES.get(mkt, RISK_FREE_RATES["IN"])
    if mkt == "MIXED":
        return fail(mmsg, "danger")

    wb = (float(min_w or 0)/100, float(max_w or 100)/100)

    try:
        prices, returns = get_data(tickers=tickers, start=start, end=end, force_refresh=True)
        n = len(prices)
        dqt, dqm = dq_banner(n)
        def pill(msg, color_key):
            col_map = {"success": C["green"], "warning": C["gold"],
                       "danger": C["red"], "info": C["primary"]}
            c = col_map.get(color_key, C["muted"])
            return html.Span(msg, style={
                "background": c + "22", "border": f"1px solid {c}55",
                "borderRadius": "20px", "color": c,
                "padding": "3px 12px", "fontSize": "0.75rem", "fontWeight": "500",
            })
        banners = html.Div([pill(mmsg, mtype), pill(dqm, dqt)],
                           style={"display":"flex","gap":"8px","flexWrap":"wrap"})
        if n < MIN_DAYS:
            return [banners] + NONE8 + [EMPTY_FIG]*6 + [[], [], None, None]

        mk  = run_markowitz(prices, risk_free_rate=rfr, weight_bounds=wb, verbose=False)
        dd  = run_drawdown(prices, returns,
                           markowitz_weights=mk["max_sharpe"]["weights"].to_dict(),
                           beta=beta, weight_bounds=wb,
                           risk_free_rate=rfr, verbose=False)
        sr  = run_shap(prices, returns, mk, dd, verbose=False)

        ms, mv, mc = mk["max_sharpe"], mk["min_vol"], dd["min_cdar"]

        sdf = mk["summary"].copy()
        sdf.loc["Min CDaR"] = {
            "Ann. Return":     pct(mc["annual_return"]),
            "Ann. Volatility": pct(mc["annual_vol"]),
            "Sharpe Ratio":    f3(mc.get("sharpe_ratio", ms["sharpe_ratio"])),
            "Top Holding":     mc["weights"].idxmax(),
            "Top Weight":      pct(mc["weights"].max()),
        }

        return (
            banners,
            pct(ms["annual_return"]),
            f"Vol {pct(ms['annual_vol'])}  ·  SR {f2(ms['sharpe_ratio'])}",
            pct(mv["annual_return"]),
            f"Vol {pct(mv['annual_vol'])}  ·  SR {f2(mv['sharpe_ratio'])}",
            pct(mc["annual_return"]),
            f"CDaR {pct(mc['cdar'])}  ·  Calmar {f2(mc['calmar_ratio'])}",
            pct(dd["markowitz_dd"]["max_drawdown"]),
            f"vs CDaR MDD {pct(mc['max_drawdown'])}",
            build_frontier(mk),
            build_cdar(dd),
            build_weights(mk, dd),
            build_drawdown(returns, mk, dd),
            build_shap_imp(sr),
            build_shap_dir(sr),
            dtable(sdf, "tbl-strategy"),
            dtable(dd["asset_table"], "tbl-asset"),
            [{"label": t, "value": t} for t in tickers],
            tickers[0],
            {   # per-ticker pre-computed waterfall data — no recompute in callback
                t: waterfall_data(sr["cdar_explainer"], sr["X"],
                                  sr["meta_df"], t, sr["feature_names"])
                for t in tickers
            },
        )

    except Exception as e:
        log.exception("Analysis failed")
        return fail(f"Error: {e}", "danger")

#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Waterfall callback - reads from shap-store, zero recompute
#-------------------------------------------------------------------------------------
@app.callback(
    Output("waterfall-chart", "figure"),
    Input("waterfall-ticker", "value"),
    Input("shap-store",       "data"),
    prevent_initial_call=True,
)
#-------------------------------------------------------------------------------------
def update_waterfall(ticker, store):
    if not ticker or not store:
        return EMPTY_FIG
    try:
        wf = store.get(ticker)
        if not wf:
            fig = go.Figure()
            fig.add_annotation(text=f"No SHAP data for {ticker}",
                               showarrow=False, font=dict(color=C["muted"], size=13))
            return fig.update_layout(**BASE())
        # Reuse the same build_waterfall rendering path — correct labels & values
        feats  = wf["features"][:10]
        svals  = wf["shap_values"][:10]
        fvals  = wf["feature_values"][:10]
        labels = [f"{f}  (val={v:.3f})" for f, v in zip(feats, fvals)]
        fig = go.Figure(go.Waterfall(
            orientation="h",
            measure=["relative"]*len(svals) + ["total"],
            x=list(svals) + [sum(svals)],
            y=labels + [f"{ticker} Prediction"],
            connector=dict(line=dict(color=C["border"], width=1)),
            increasing=dict(marker=dict(color=C["green"])),
            decreasing=dict(marker=dict(color=C["red"])),
            totals=dict(marker=dict(color=C["primary"])),
            text=[f"{v:+.5f}" for v in svals] + [f"{sum(svals):+.5f}"],
            textposition="outside", textfont=dict(color=C["text"], size=10),
        ))
        fig.update_layout(**BASE(margin=dict(l=210, r=40, t=50, b=50)),
            legend=LEG,
            title=dict(
                text=(f"SHAP Waterfall — {ticker}  |  "
                      f"Base: {wf['base_value']:.4f}  |  Date: {str(wf['date'])[:10]}"),
                font=dict(size=11, color=C["muted"]), x=0),
            xaxis=dict(**AX, title="SHAP Value (impact on predicted weight)"),
            yaxis=dict(**AX, autorange="reversed"))
        return fig
    except Exception as e:
        log.exception("Waterfall (store) failed")
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {e}", showarrow=False,
                           font=dict(color=C["red"], size=12))
        return fig.update_layout(**BASE())

#-------------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=8050)

#-------------------------------------------------------------------------------------