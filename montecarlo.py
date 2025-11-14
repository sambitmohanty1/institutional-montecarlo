# montecarlo.py
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Helper: fetch historical returns
# ---------------------------------------------------------
def get_returns(ticker, period="3y"):
    data = yf.Ticker(ticker).history(period=period)
    if data.empty:
        raise ValueError(f"No data for ticker {ticker}")
    ret = data["Close"].pct_change().dropna()
    return data, ret


# ---------------------------------------------------------
# 1. Single-Asset GBM Monte Carlo
# ---------------------------------------------------------
def run_montecarlo_simulation(ticker, years=3, simulations=1000):
    """
    Classical Geometric Brownian Motion Monte Carlo for a single stock.
    Includes:
    - drift estimated from historical mean
    - volatility from daily returns
    - dividends reinvested (approx via total-return adjustment)
    Returns: dict with 'plot', 'expected_cagr', 'percentiles'
    """
    hist, returns = get_returns(ticker)

    price0 = hist["Close"].iloc[-1]

    mu = returns.mean() * 252  # annualized drift
    sigma = returns.std() * np.sqrt(252)  # annualized vol

    dy = yf.Ticker(ticker).info.get("dividendYield", 0) or 0
    div_factor = 1 + dy

    days = int(252 * years)
    dt = 1 / 252

    paths = np.zeros((simulations, days))
    paths[:, 0] = price0

    for t in range(1, days):
        z = np.random.standard_normal(simulations)
        paths[:, t] = paths[:, t - 1] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        )

    # Apply dividend reinvestment
    paths = paths * div_factor

    # Summaries
    final_prices = paths[:, -1]
    percentiles = {
        "5%": round(np.percentile(final_prices, 5), 2),
        "25%": round(np.percentile(final_prices, 25), 2),
        "50%": round(np.percentile(final_prices, 50), 2),
        "75%": round(np.percentile(final_prices, 75), 2),
        "95%": round(np.percentile(final_prices, 95), 2),
    }

    median_final = percentiles["50%"]
    cagr = (median_final / price0) ** (1 / years) - 1
    cagr_str = f"{cagr * 100:.2f}%"

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    px = np.percentile(paths, [10, 50, 90], axis=0)
    ax.plot(px[1], label="Median", linewidth=2)
    ax.fill_between(range(days), px[0], px[2], alpha=0.2, label="10–90%")
    ax.set_title(f"{ticker} Monte Carlo Projection ({simulations} sims)")
    ax.set_xlabel("Days")
    ax.set_ylabel("Price")
    ax.legend()

    return {
        "plot": fig,
        "expected_cagr": cagr_str,
        "percentiles": percentiles,
    }


# ---------------------------------------------------------
# 2. FULL PORTFOLIO MONTE CARLO (CORRELATED)
# ---------------------------------------------------------
def run_portfolio_montecarlo(tickers, weights, years=3, sims=5000):
    """
    Institutional-grade portfolio Monte Carlo:
    - fetches all tickers' price history
    - builds correlation + covariance matrix
    - uses Cholesky to generate correlated returns
    - simulates entire portfolio path
    - reinvests all dividends
    - returns dict with plot, percentiles, expected CAGR, summary table
    """

    # Fetch historical data for all tickers
    price_data = {}
    return_data = {}

    for t in tickers:
        hist, ret = get_returns(t)
        price_data[t] = hist["Close"]
        return_data[t] = ret

    # Align all return series
    df_returns = pd.DataFrame(return_data).dropna()

    mu_vec = df_returns.mean().values * 252  # annualized drift
    cov = df_returns.cov().values * 252  # annualized covariance

    chol = np.linalg.cholesky(cov)

    div_yields = []
    for t in tickers:
        dy = yf.Ticker(t).info.get("dividendYield", 0) or 0
        div_yields.append(1 + dy)

    N = sims
    days = int(252 * years)
    dt = 1 / 252

    portfolio_paths = np.zeros((N, days))
    portfolio_paths[:, 0] = sum(
        w * price_data[t].iloc[-1] for w, t in zip(weights, tickers)
    )

    # Initial prices vector
    p0_vec = np.array([price_data[t].iloc[-1] for t in tickers])

    for s in range(N):
        prices = p0_vec.copy()
        for d in range(1, days):
            z = np.random.standard_normal(len(tickers))
            correlated_shock = chol @ z

            prices = prices * np.exp(
                (mu_vec - 0.5 * np.diag(cov)) * dt
                + np.sqrt(np.diag(cov)) * correlated_shock * np.sqrt(dt)
            )

            # apply dividends
            prices = prices * div_yields

            portfolio_paths[s, d] = np.dot(weights, prices)

    final_vals = portfolio_paths[:, -1]
    initial_portfolio = portfolio_paths[:, 0]

    percentiles = {
        "5%": round(np.percentile(final_vals, 5), 2),
        "25%": round(np.percentile(final_vals, 25), 2),
        "50%": round(np.percentile(final_vals, 50), 2),
        "75%": round(np.percentile(final_vals, 75), 2),
        "95%": round(np.percentile(final_vals, 95), 2),
    }

    median_final = percentiles["50%"]
    cagr = (median_final / initial_portfolio.mean()) ** (1 / years) - 1
    cagr_str = f"{cagr * 100:.2f}%"

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    px = np.percentile(portfolio_paths, [10, 50, 90], axis=0)
    ax.plot(px[1], label="Median", linewidth=2)
    ax.fill_between(range(days), px[0], px[2], alpha=0.25)
    ax.set_title(f"Portfolio Monte Carlo — {sims} simulations")
    ax.set_xlabel("Days")
    ax.set_ylabel("Portfolio Value")
    ax.legend()

    summary_table = pd.DataFrame({
        "Ticker": tickers,
        "Weight": weights,
        "DivYield": div_yields,
        "Ann_Return": mu_vec,
    })

    return {
        "plot": fig,
        "expected_cagr": cagr_str,
        "percentiles": percentiles,
        "summary_table": summary_table,
    }
