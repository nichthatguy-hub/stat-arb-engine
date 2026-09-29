import os
import tempfile
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import yfinance as yf

# Point yfinance cache to a clean temporary system directory
temp_cache_dir = os.path.join(tempfile.gettempdir(), "yfinance_cache")
os.makedirs(temp_cache_dir, exist_ok=True)
yf.set_tz_cache_location(temp_cache_dir)

def fetch_pair_data(ticker1: str, ticker2: str, start_date: str, end_date: str) -> pd.DataFrame:
    print(f"Downloading historical daily data for {ticker1} and {ticker2}...")
    
    # Download using yf.download with multi_level_index=False
    raw = yf.download(
        tickers=[ticker1, ticker2],
        start=start_date,
        end=end_date,
        progress=False
    )
    
    # Extract closing prices
    close_prices = raw['Close']
    df = close_prices[[ticker1, ticker2]].dropna()
    
    print(f"Successfully loaded {len(df)} trading days of data.")
    return df

def test_cointegration(df: pd.DataFrame, ticker1: str, ticker2: str):
    # Step A: OLS Regression -> P1 = alpha + beta * P2 + epsilon
    y = df[ticker1]
    X = sm.add_constant(df[ticker2])
    ols_model = sm.OLS(y, X).fit()
    
    alpha = ols_model.params['const']
    beta = ols_model.params[ticker2]
    
    # Step B: Compute the residual spread
    spread = y - (alpha + beta * df[ticker2])
    
    # Step C: Augmented Dickey-Fuller Test on the residuals
    adf_result = adfuller(spread)
    adf_stat = adf_result[0]
    p_val = adf_result[1]
    crit_vals = adf_result[4]
    
    return beta, alpha, adf_stat, p_val, crit_vals

if __name__ == "__main__":
    # Choose any pair from above, e.g., Lam Research vs. Applied Materials
    t1, t2 = "SPGI", "MCO"
    prices = fetch_pair_data(t1, t2, "2020-01-01", "2024-01-01")
    
    beta, alpha, adf_stat, p_val, crit_vals = test_cointegration(prices, t1, t2)
    
    print("\n" + "=" * 55)
    print(f"Target Pair:           {t1} and {t2}")
    print(f"Hedge Ratio (Beta):    {beta:.4f}")
    print(f"Intercept (Alpha):     {alpha:.4f}")
    print(f"ADF Test Statistic:    {adf_stat:.4f}")
    print(f"ADF p-value:           {p_val:.4f}")
    print(f"5% Critical Threshold: {crit_vals['5%']:.3f}")
    print("=" * 55)
    
    if p_val < 0.05:
        print("✓ PASS: The pair is cointegrated (p < 0.05). Ready to trade!")
    else:
        print("✗ FAIL: The pair is not cointegrated (p >= 0.05).")