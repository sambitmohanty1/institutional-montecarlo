import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from montecarlo import run_montecarlo_simulation

def generate_investment_thesis(ticker, info, technicals, mc):
    price = technicals["price"]
    sma50 = technicals["sma50"]
    sma200 = technicals["sma200"]

    trend = "bullish" if sma50[-1] > sma200[-1] else "bearish"

    sector = info.get("sector", "N/A")
    pe = info.get("trailingPE", None)
    div = info.get("dividendYield", 0) * 100
    market_cap = info.get("marketCap", 0) / 1e9

    expected_cagr = mc["expected_cagr"]

    thesis = f"""
### 🧠 AI-Generated Investment Thesis for **{ticker}**

**Technical Trend:**  
The stock is currently showing a *{trend}* setup based on the 50/200 SMA crossover pattern.  
This generally indicates that momentum is shifting towards the {trend} side.

**Fundamentals:**  
- Market Cap: **${market_cap:.2f}B**  
- P/E Ratio: **{pe if pe else "N/A"}**  
- Dividend Yield: **{div:.2f}%**  
- Sector: **{sector}**

The valuation appears {'reasonable' if pe and pe < 30 else 'elevated'} relative to sector norms, indicating that investors are pricing in {'growth potential' if pe and pe > 30 else 'stable earnings'}.

**Monte Carlo Forward View (3-Year Outlook):**  
Expected CAGR: **{expected_cagr}**  
The distribution indicates a balanced risk-reward profile with healthy upside scenarios.

**Overall Thesis:**  
{ticker} appears to provide a combination of { 'growth and momentum' if trend == 'bullish' else 'value and stability'}.  
Given its sector dynamics, expected return profile, and technical structure, the stock may be suitable for investors with a **{ 'growth-oriented' if trend == 'bullish' else 'risk-aware' }** outlook over the next 2–4 years.

---

### 🟢 Investment Outlook:  
**Moderately Positive** if the investor seeks multi-year growth and can tolerate moderate volatility.
"""
    return thesis

st.set_page_config(page_title="Institutional Monte Carlo + Stock Analyst", layout="wide")

st.title("🏦 Institutional Monte Carlo + AI Stock Analyst")
st.markdown("Analyze stocks (US / ASX) and simulate 3-year portfolio performance with dividend reinvestment.")

mode = st.sidebar.selectbox("Choose Mode", ["Single Stock", "Portfolio Simulation"])

if mode == "Single Stock":
    ticker = st.text_input("Enter stock ticker (e.g. MSFT, META, CSL.AX, XRO.AX):")
    if ticker:
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="1y")
            info = stock.info
            price = data['Close'].iloc[-1]
            sma50 = data['Close'].rolling(50).mean()
            sma200 = data['Close'].rolling(200).mean()

            st.subheader(f"📈 {ticker} — Technical Overview")
            fig, ax = plt.subplots()
            ax.plot(data.index, data['Close'], label='Price', linewidth=1.5)
            ax.plot(data.index, sma50, label='SMA50')
            ax.plot(data.index, sma200, label='SMA200')
            ax.legend()
            ax.set_title(f"{ticker} - 1 Year Chart")
            st.pyplot(fig)

            st.subheader("📊 Key Fundamentals")
            st.markdown(f"""
            **Current Price:** ${price:.2f}  
            **Market Cap:** {info.get('marketCap', 0)/1e9:.2f} B  
            **P/E Ratio:** {info.get('trailingPE', 'N/A')}  
            **Dividend Yield:** {info.get('dividendYield', 0)*100:.2f}%  
            **Sector:** {info.get('sector', 'N/A')}  
            """)

            # Monte Carlo projection
            st.subheader("🎲 Monte Carlo Projection (3 Years)")
            mc_result = run_montecarlo_simulation(ticker, years=3, simulations=500)
            st.pyplot(mc_result['plot'])
            st.write("Expected CAGR:", mc_result['expected_cagr'])
            st.write("5th–95th percentile:", mc_result['percentiles'])

        except Exception as e:
            st.error(f"Error fetching data: {e}")

else:
    st.subheader("📁 Portfolio Monte Carlo Simulation")
    uploaded_file = st.file_uploader("Upload CSV (ticker,weight)", type=['csv'])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)
        results = []
        for _, row in df.iterrows():
            try:
                r = run_montecarlo_simulation(row['ticker'], years=3, simulations=300)
                results.append({
                    "ticker": row['ticker'],
                    "expected_cagr": r['expected_cagr']
                })
            except:
                pass
        st.subheader("Portfolio Simulation Summary")
        st.dataframe(pd.DataFrame(results))

# Generate AI thesis
technicals = {
    "price": price,
    "sma50": sma50,
    "sma200": sma200
}

ai_thesis = generate_investment_thesis(ticker, info, technicals, mc_result)

st.subheader("🧠 AI Investment Thesis")
st.markdown(ai_thesis)

