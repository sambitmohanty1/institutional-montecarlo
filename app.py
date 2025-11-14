# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from montecarlo import run_montecarlo_simulation, run_portfolio_montecarlo

# ---------------------------
# Helper: AI Investment Thesis
# ---------------------------
def generate_investment_thesis(ticker, info, technicals, mc):
    price = technicals.get("price", None)
    sma50 = technicals.get("sma50", None)
    sma200 = technicals.get("sma200", None)

    # Determine trend safely
    try:
        trend = "bullish" if sma50.iloc[-1] > sma200.iloc[-1] else "bearish"
    except Exception:
        trend = "neutral"

    sector = info.get("sector", "N/A")
    pe = info.get("trailingPE", None)
    dy = info.get("dividendYield", 0) or 0
    div = dy * 100
    market_cap = info.get("marketCap", 0) / 1e9 if info.get("marketCap") else None

    expected_cagr = mc.get("expected_cagr", "N/A")
    percentiles = mc.get("percentiles", {})

    if not pe:
        val_text = "Valuation (P/E) unavailable."
    elif pe < 15:
        val_text = f"P/E {pe:.1f} = undervalued historically."
    elif pe < 30:
        val_text = f"P/E {pe:.1f} = fairly valued."
    else:
        val_text = f"P/E {pe:.1f} = priced for high growth."

    thesis = f"""
### 🧠 AI Investment Thesis: **{ticker}**

**Technical Trend:**  
- 50-day vs 200-day SMA → **{trend}**

**Fundamentals:**  
- **Sector:** {sector}  
- **Market Cap:** {'${:,.2f}B'.format(market_cap) if market_cap else 'N/A'}  
- **P/E:** {pe}  
- **Dividend Yield:** {div:.2f}%  

{val_text}

**Forward-Looking (Monte Carlo, 3 years):**  
- Expected CAGR: **{expected_cagr}**  
- Percentile distribution: **{percentiles}**

**Interpretation:**  
{ticker} fits a **{'growth-oriented' if trend=='bullish' else 'risk-sensitive' if trend=='bearish' else 'balanced'}** profile.  
Suitable for investors with a **2–4 year horizon**.

**Risks:** macro shocks, regulation, earnings volatility, and sentiment swings.
"""
    return thesis


# ---------------------------
# Streamlit Layout
# ---------------------------
st.set_page_config(page_title="Institutional Monte Carlo + AI Analyst", layout="wide")
st.title("🏦 Institutional Monte Carlo + AI Stock Analyst")

mode = st.sidebar.selectbox("Mode", ["Single Stock", "Portfolio Simulation"])


# ============================
#  SINGLE STOCK MODE
# ============================
if mode == "Single Stock":
    st.header("Single Stock Analysis")

    ticker = st.text_input(
        "Enter stock ticker (US or ASX using .AX suffix):",
        value="MSFT"
    ).strip().upper()

    if st.button("Run Analysis"):

        with st.spinner(f"Loading data for {ticker}..."):
            try:
                yf_ticker = yf.Ticker(ticker)
                hist = yf_ticker.history(period="1y")

                if hist.empty:
                    st.error("No historical data found. Check the ticker.")
                    st.stop()

                # FIX: Define price *before* referencing later
                price = hist["Close"].iloc[-1]

                # Technicals
                sma50 = hist["Close"].rolling(50).mean()
                sma200 = hist["Close"].rolling(200).mean()

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(hist["Close"], label="Close")
                ax.plot(sma50, label="SMA50")
                ax.plot(sma200, label="SMA200")
                ax.legend()
                ax.set_title(f"{ticker} Price + SMAs")
                st.pyplot(fig)

                # Fundamentals
                info = yf_ticker.info or {}

                market_cap = info.get("marketCap", None)
                pe = info.get("trailingPE", None)
                dy = info.get("dividendYield", 0) or 0
                sector = info.get("sector", "N/A")
                target = info.get("targetMeanPrice", None)

                st.subheader("Fundamentals")
                st.write(f"**Price:** ${price:.2f}")
                st.write(f"**Sector:** {sector}")
                st.write(
                    f"**Market Cap:** ${market_cap/1e9:.2f}B"
                    if market_cap else "Market Cap unavailable"
                )
                st.write(f"**P/E:** {pe if pe else 'N/A'}")
                st.write(f"**Dividend Yield:** {dy*100:.2f}%")
                if target:
                    st.write(
                        f"**Analyst Target:** ${target:.2f} "
                        f"(Upside {(target/price-1)*100:.1f}%)"
                    )

                # Sentiment (simple)
                last30 = hist["Close"].pct_change().tail(30).add(1).prod() - 1
                sentiment = "Positive" if last30 > 0 else "Negative" if last30 < 0 else "Neutral"
                st.write(f"**30-day sentiment:** {sentiment} ({last30*100:.2f}%)")

                # Monte Carlo simulation
                st.subheader("Monte Carlo Simulation")
                years = st.slider("Projection horizon (years)", 1, 10, 3)
                sims = st.slider("Simulations", 500, 5000, 2000, step=500)

                mc = run_montecarlo_simulation(ticker, years, sims)
                st.pyplot(mc["plot"])
                st.write(f"**Expected CAGR:** {mc['expected_cagr']}")
                st.write("**Percentiles:**", mc["percentiles"])

                # AI thesis
                st.subheader("AI Investment Thesis")
                technicals = {"price": price, "sma50": sma50, "sma200": sma200}
                thesis = generate_investment_thesis(ticker, info, technicals, mc)
                st.markdown(thesis)

            except Exception as e:
                st.error(f"Error: {e}")


# ============================
#  PORTFOLIO MODE
# ============================
else:
    st.header("Portfolio Monte Carlo (Correlated)")

    st.markdown("---")
Upload a CSV file containing:

