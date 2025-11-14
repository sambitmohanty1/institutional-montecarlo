# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from montecarlo import run_montecarlo_simulation

# ---------------------------
# Helper: AI Investment Thesis
# ---------------------------
def generate_investment_thesis(ticker, info, technicals, mc):
    """
    Create a concise, readable AI-style investment thesis.
    - info: dict from yfinance .info
    - technicals: dict with price, sma50 (Series), sma200 (Series)
    - mc: dict returned from run_montecarlo_simulation
    """
    price = technicals.get("price", None)
    sma50 = technicals.get("sma50", None)
    sma200 = technicals.get("sma200", None)

    # Determine trend
    try:
        trend = "bullish" if sma50.iloc[-1] > sma200.iloc[-1] else "bearish"
    except Exception:
        trend = "neutral"

    sector = info.get("sector", "N/A")
    pe = info.get("trailingPE", None)
    dy = info.get("dividendYield", 0) or 0.0
    div = dy * 100
    market_cap = info.get("marketCap", 0) / 1e9 if info.get("marketCap") else None

    expected_cagr = mc.get("expected_cagr", "N/A") if mc else "N/A"
    percentiles = mc.get("percentiles", {}) if mc else {}

    # Valuation description
    if pe is None:
        val_text = "Valuation data (P/E) is not available."
    elif pe < 15:
        val_text = f"P/E ~{pe:.1f}, valuation is attractive relative to historical norms."
    elif pe < 30:
        val_text = f"P/E ~{pe:.1f}, valuation appears fair—buy on dips for medium-term horizon."
    else:
        val_text = f"P/E ~{pe:.1f}, valuation is elevated and priced for strong growth."

    thesis = f"""
### 🧠 AI-Generated Investment Thesis for **{ticker}**

**Technical Trend:**  
The short/medium term technical structure is **{trend}** (50-day vs 200-day SMA).

**Fundamentals snapshot:**  
- **Sector:** {sector}  
- **Market Cap:** {'${:,.2f}B'.format(market_cap) if market_cap else 'N/A'}  
- **P/E Ratio:** {pe if pe else 'N/A'}  
- **Dividend Yield:** {div:.2f}%  

{val_text}

**Monte Carlo (3-year) forward view:**  
- Expected CAGR (median estimate): **{expected_cagr}**  
- Key percentiles (final price distribution): **{percentiles}**

**Investment view:**  
{ticker} combines {('momentum' if trend=='bullish' else 'value/stability' if trend=='bearish' else 'mixed signals')}.  Based on the simulated forward paths and fundamentals, the stock is suited to investors who seek **{'growth-oriented returns' if trend=='bullish' else 'a more risk-aware allocation' }** over a 2–4 year horizon.

**Risks to watch:** macro shocks, earnings misses, sector-specific regulation, and sudden volatility spikes. Always re-evaluate after quarterly earnings or significant news.

"""
    return thesis

# ---------------------------
# Streamlit App
# ---------------------------
st.set_page_config(page_title="Institutional Monte Carlo + Stock Analyst", layout="wide")
st.title("🏦 Institutional Monte Carlo + AI Stock Analyst")
st.markdown("Analyze US & ASX stocks and simulate 3-year portfolio performance with dividend-aware Monte Carlo projections.")

mode = st.sidebar.selectbox("Mode", ["Single Stock", "Portfolio Simulation"])

# ---------------------------
# SINGLE STOCK MODE
# ---------------------------
if mode == "Single Stock":
    st.header("Single Stock Analysis")
    ticker = st.text_input("Enter stock ticker (US or ASX — use .AX for ASX):", value="MSFT").strip().upper()
    col1, col2 = st.columns([2, 1])

    if st.button("Run Analysis") and ticker:
        with st.spinner(f"Fetching data for {ticker} ..."):
            try:
                # Fetch data
                yf_ticker = yf.Ticker(ticker)
                hist = yf_ticker.history(period="1y", interval="1d").dropna()
                if hist.empty:
                    st.error("No historical price data found. Check ticker symbol (use .AX suffix for ASX).")
                else:
                    info = yf_ticker.info or {}

                    # Technicals
                    price = hist['Close'].iloc[-1]
                    sma50 = hist['Close'].rolling(50).mean()
                    sma200 = hist['Close'].rolling(200).mean()

                    # Plot price + SMAs
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(hist.index, hist['Close'], label='Close', linewidth=1.2)
                    ax.plot(hist.index, sma50, label='SMA50', linewidth=1.0)
                    ax.plot(hist.index, sma200, label='SMA200', linewidth=1.0)
                    ax.set_title(f"{ticker} — 1 Year Price + SMA50/200")
                    ax.legend()
                    st.pyplot(fig)

                    # Fundamentals display
                    market_cap = info.get('marketCap', None)
                    pe = info.get('trailingPE', None)
                    div_yield = info.get('dividendYield', 0) or 0.0
                    sector = info.get('sector', 'N/A')
                    target = info.get('targetMeanPrice', None)
                    analyst_upside = None
                    if target and price:
                        analyst_upside = (target - price) / price * 100

                    st.subheader("Key Fundamentals")
                    st.write(f"**Price:** ${price:.2f}")
                    st.write(f"**Market Cap:** {market_cap/1e9:.2f} B" if market_cap else "**Market Cap:** N/A")
                    st.write(f"**P/E (trailing):** {pe if pe else 'N/A'}")
                    st.write(f"**Dividend Yield:** {div_yield*100:.2f}%")
                    st.write(f"**Sector:** {sector}")
                    if target:
                        st.write(f"**Analyst target (mean):** ${target:.2f}  → upside: {analyst_upside:.1f}%")

                    # Sentiment (simple)
                    last30 = hist['Close'].pct_change().tail(30).add(1).prod() - 1
                    sentiment = "Positive" if last30 > 0 else "Negative" if last30 < 0 else "Neutral"
                    st.write(f"**30-day return:** {last30*100:.2f}% → Sentiment: **{sentiment}**")

                    # Monte Carlo projection (single asset)
                    st.subheader("Monte Carlo Projection (total-return approx.)")
                    years = st.slider("Projection years", 1, 10, 3, key="mc_years")
                    sims = st.slider("Number of simulations", 200, 5000, 1000, step=200, key="mc_sims")
                    mc_result = run_montecarlo_simulation(ticker, years=years, simulations=sims)

                    # display plot returned by montecarlo function
                    try:
                        st.pyplot(mc_result['plot'])
                    except Exception:
                        st.info("Plot could not be rendered from simulation. You can still view text outputs.")

                    st.write("**Expected CAGR (median):**", mc_result.get('expected_cagr', 'N/A'))
                    st.write("**Distribution percentiles (final prices):**", mc_result.get('percentiles', {}))

                    # AI thesis
                    technicals = {"price": price, "sma50": sma50, "sma200": sma200}
                    ai_text = generate_investment_thesis(ticker, info, technicals, mc_result)
                    st.subheader("AI Investment Thesis")
                    st.markdown(ai_text)

            except Exception as e:
                st.error(f"Error during analysis: {e}")

# ---------------------------
# PORTFOLIO SIMULATION MODE
# ---------------------------
else:
    st.header("Portfolio Monte Carlo Simulation")
    st.markdown("Upload a CSV with `ticker,weight` columns (weights as decimals summing to 1 or as percentages). Or use manual entry below.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    manual = st.text_area("Or paste tickers & weights (one per line, e.g. MSFT,0.4):", height=120)

    tickers = []
    weights = []

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            if 'ticker' not in df.columns or 'weight' not in df.columns:
                st.error("CSV must have 'ticker' and 'weight' columns.")
            else:
                tickers = df['ticker'].astype(str).str.strip().str.upper().tolist()
                weights = df['weight'].astype(float).tolist()
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

    elif manual:
        lines = [l.strip() for l in manual.splitlines() if l.strip()]
        for ln in lines:
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) >= 1:
                t = parts[0].upper()
                w = float(parts[1]) if len(parts) > 1 else None
                tickers.append(t)
                weights.append(w)

    if tickers:
        # Normalize weights: if any are None or sum to ~0, use equal weights
        try:
            weights = [float(w) if w is not None else 0 for w in weights]
            if sum(weights) == 0:
                weights = [1.0 / len(tickers)] * len(tickers)
            else:
                total = sum(weights)
                weights = [w / total for w in weights]
        except Exception:
            weights = [1.0 / len(tickers)] * len(tickers)

        st.write("Tickers:", tickers)
        st.write("Weights:", [round(w, 3) for w in weights])

        # Run simulation per ticker and aggregate summary
        years = st.slider("Projection years", 1, 10, 3, key="port_years")
        sims = st.slider("Simulations per ticker", 200, 3000, 1000, step=200, key="port_sims")

        results = []
        progress = st.progress(0)
        for i, t in enumerate(tickers):
            try:
                res = run_montecarlo_simulation(t, years=years, simulations=sims)
                # expected_cagr is a string like "12.34%", ensure numeric
                exp_cagr_str = res.get('expected_cagr', '0%')
                try:
                    exp_cagr_val = float(str(exp_cagr_str).strip('%'))  # percent
                except Exception:
                    exp_cagr_val = None
                results.append({
                    "ticker": t,
                    "weight": weights[i],
                    "expected_cagr_%": exp_cagr_val,
                    "percentiles": res.get('percentiles', {})
                })
            except Exception as e:
                results.append({"ticker": t, "weight": weights[i], "error": str(e)})
            progress.progress((i + 1) / len(tickers))

        # Build DataFrame
        df_res = pd.DataFrame(results)
        st.subheader("Per-asset simulation summary")
        st.dataframe(df_res)

        # Portfolio-level simple aggregation: weighted average CAGR
        # (Note: true portfolio simulation would simulate correlated paths — this is a quick heuristic)
        valid_cagrs = [r["expected_cagr_%"] for r in results if r.get("expected_cagr_%") is not None]
        if valid_cagrs:
            weighted_cagr = sum((r["expected_cagr_%"] or 0) * r["weight"] for r in results)  # in percent
            st.write(f"Weighted average expected CAGR (heuristic): **{weighted_cagr:.2f}%**")

        st.info("Note: This portfolio mode runs independent simulations per asset and aggregates heuristically. For a joint correlated portfolio simulation, replace with a covariance-based portfolio Monte Carlo model (future enhancement).")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("This tool is for modeling and educational use only — not financial advice.")
