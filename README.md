# Portfolio Optimisation with Drawdown Constraints and XAI

Quantitative portfolio optimisation project comparing **Markowitz Mean-Variance** and **CDaR (Conditional Drawdown-at-Risk)** portfolio construction, with **SHAP explainability** to interpret stock-level allocation decisions.

## Project Overview

This project builds an end-to-end research and dashboard pipeline for portfolio construction on Indian banking equities.

It combines:

- **Markowitz optimisation** for mean-variance efficient portfolios
- **CDaR optimisation** for downside-risk-aware allocation
- **XGBoost + SHAP** for explainable portfolio weight attribution
- **Plotly Dash dashboard** for interactive exploration

The goal is not only to generate portfolio weights, but also to explain **why** particular assets receive higher or lower allocations under different risk frameworks.

## Market Setup

- **Market:** India (NSE/BSE)
- **Universe:** AXISBANK.NS, HDFCBANK.NS, ICICIBANK.NS, KOTAKBANK.NS, PNB.NS, SBIN.NS
- **Period:** 2015-01-01 to 2026-05-24
- **Risk-free rate:** RBI Repo Rate (5.25%)
- **CDaR beta:** 0.95
- **Weight constraints:** min 6%, max 36%

## Features

### 1. Portfolio Construction
- Maximum Sharpe portfolio
- Minimum volatility portfolio
- Minimum CDaR portfolio
- Equal-weight benchmark

### 2. Risk Analytics
- Efficient frontiers
- Underwater drawdown curves
- Per-asset drawdown statistics
- Strategy comparison tables

### 3. Explainable AI Layer
- XGBoost surrogate model for weight prediction
- SHAP global feature importance
- Signed SHAP direction heatmap
- Per-ticker SHAP waterfall explanation

### 4. Dashboard
Interactive Dash app with 5 tabs:
- Overview
- Frontiers
- Drawdown
- SHAP
- Deep Dive

## Repository Structure

```bash
portfolio-optimisation-drawdown-xai/
│
├── data/
│   └── fetch_data.py
│
├── models/
│   ├── markowitz.py
│   ├── drawdown.py
│   └── shap_attribution.py
│
├── dashboard/
│   └── app.py
│
├── notebooks/
│   └── research notebooks
│
└── README.md
```

## Example Insights

- CDaR optimisation focuses directly on drawdown risk rather than only volatility.
- The same feature can affect assets differently; for example, `volatility_long` may reduce one stock’s weight while increasing another’s.
- SHAP makes the optimiser auditable by decomposing each predicted allocation into feature-level contributions.

## Tech Stack

- Python
- pandas
- numpy
- yfinance
- PyPortfolioOpt
- xgboost
- shap
- plotly
- dash

## How to Run

### 1. Clone the repo
```bash
git clone <repo-url>
cd portfolio-optimisation-drawdown-xai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the dashboard
```bash
python dashboard/app.py
```

Then open:

```bash
http://127.0.0.1:8050
```

## Why this project matters

Traditional portfolio optimisation often behaves like a black box: weights come out, but the reasons remain unclear.

This project adds an explainability layer on top of portfolio construction, making allocation decisions more interpretable for research, risk management, and financial AI applications.

## Author

**Shrutik Panchal**

Quantitative finance / financial AI project focused on portfolio optimisation, downside-risk modelling, and explainable machine learning.