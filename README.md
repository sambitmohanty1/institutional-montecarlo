# Institutional Monte Carlo + Stock Analyst (Consolidated)

This Streamlit app merges a single-ticker analyst dashboard with a Monte Carlo portfolio simulator that models dividend reinvestment.

## Features
- Single ticker analysis: SMA50/200, fundamentals, analyst target, quick Monte Carlo (single asset).
- Portfolio mode: upload CSV (`ticker,weight`) or manual list, runs Monte Carlo for portfolio with dividend reinvestment, returns percentiles, CAGR estimates, and download CSV of results.
- Plots use matplotlib.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
1. Push to GitHub (your repo: institutional-montecarlo).
2. Go to https://share.streamlit.io and connect the repo.
3. Deploy `app.py` from branch `main`.

## Notes
- This is a modeling tool. Monte Carlo assumes geometric Brownian motion using historical mean/volatility.
- Dividend yields are taken from yfinance (if available). For more accurate dividend modeling, replace with company-specific payout schedules.
