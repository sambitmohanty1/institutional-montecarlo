import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from montecarlo import run_montecarlo_simulation

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
