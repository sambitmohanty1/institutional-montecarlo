import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

def run_montecarlo_simulation(ticker, years=3, simulations=500):
    data = yf.Ticker(ticker).history(period="3y")['Close']
    daily_returns = data.pct_change().dropna()
    mu, sigma = daily_returns.mean(), daily_returns.std()

    start_price = data[-1]
    trading_days = int(252 * years)
    simulated_prices = np.zeros((trading_days, simulations))

    for i in range(simulations):
        price_series = [start_price]
        for _ in range(trading_days):
            price_series.append(price_series[-1] * (1 + np.random.normal(mu, sigma)))
        simulated_prices[:, i] = price_series[1:]

    fig, ax = plt.subplots()
    ax.plot(simulated_prices, alpha=0.1, color='gray')
    ax.set_title(f"{ticker} Monte Carlo Simulation ({years} Years)")
    ax.set_xlabel("Days")
    ax.set_ylabel("Price")

    final_prices = simulated_prices[-1, :]
    expected_cagr = ((np.mean(final_prices) / start_price) ** (1/years)) - 1
    percentiles = np.percentile(final_prices, [5, 50, 95])

    return {
        "plot": fig,
        "expected_cagr": f"{expected_cagr*100:.2f}%",
        "percentiles": { "5%": round(percentiles[0], 2), "50%": round(percentiles[1], 2), "95%": round(percentiles[2], 2) }
    }
