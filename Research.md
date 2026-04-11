# Research Statement

## Interpretable Drawdown-Constrained Portfolio Optimisation

**Shrutik Panchal** | Financial AI Research Portfolio

---

Modern portfolio theory, as formalised by Markowitz (1952), remains the dominant framework for institutional asset allocation. Yet it carries two well-documented limitations that this project directly addresses: its exclusive reliance on volatility as a risk measure, which systematically underestimates losses during prolonged market drawdowns; and its complete opacity, producing allocation weights with no explanation of which risk factors or asset characteristics drove the decision.

The first limitation is practically significant. During the 2020 COVID crash and the 2022 global rate hike cycle, volatility-minimising portfolios suffered drawdown depths far exceeding their ex-ante risk estimates, because standard deviation treats upside and downside deviations symmetrically. Conditional Drawdown-at-Risk (CDaR), formalised by Chekhlov, Uryasev and Zabarankin (2005), addresses this directly: it constrains the *expected depth of drawdowns* in the worst scenarios, producing portfolios that are empirically more resilient during stress periods at a modest cost to bull-market performance.

The second limitation is increasingly regulatory. The EU Artificial Intelligence Act (2024) designates automated portfolio management systems as high-risk AI applications subject to Article 13 transparency requirements. SEBI's 2023 algorithmic trading guidelines similarly mandate explainability for automated investment decisions in Indian markets. Current institutional tools — Bloomberg PORT, MSCI RiskMetrics — compute optimal weights but offer no mechanism to explain *why* a specific allocation was recommended. SHAP (SHapley Additive exPlanations; Lundberg & Lee, 2017) provides a theoretically grounded, model-agnostic framework to decompose any allocation decision into per-asset and per-factor contributions, satisfying this regulatory gap.

This project constructs a rigorous empirical comparison of three optimisation methodologies — Markowitz MVO, CVaR Minimisation, and CDaR-Constrained Optimisation — across multiple market regimes using NSE large-cap and global equity data sourced via yfinance (2020–2026). Walk-forward backtesting isolates regime-specific performance differences. A SHAP attribution layer is then applied to each model's allocation decisions, generating interpretable explanations that identify which assets and macro-risk factors most influence the optimal weight assignment at each rebalancing point.

The central hypothesis is that CDaR-constrained models will demonstrate statistically significant drawdown reduction during stress regimes without proportional sacrifice of risk-adjusted returns, and that SHAP attribution will reveal systematic differences in the factor exposures each model implicitly targets — differences invisible to weight-level analysis alone.

---

*This project is part of a broader research portfolio in Interpretable ML for Financial Risk Under Uncertainty.*
