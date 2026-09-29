import os
import tempfile
import numpy as np
import pandas as pd
import yfinance as yf

# Official Alpaca SDK imports
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# --- CONFIGURATION ---
API_KEY = "PKASC22YBPRYCO724YIMSIKVPD"
SECRET_KEY = "ECVgqiWqAVvDdJRNa5g5HbsBbCe4eXLwW8gqXKkxy3W5"

T1 = "SPGI"
T2 = "MCO"
BETA = 1.0426
LOOKBACK = 30
Z_ENTRY = 2.0
Z_EXIT = 0.5
EQUITY_ALLOCATION_PCT = 0.35  # 35% per leg = 70% gross exposure, 30% cash buffer

# System cache setup
temp_cache_dir = os.path.join(tempfile.gettempdir(), "yfinance_cache")
os.makedirs(temp_cache_dir, exist_ok=True)
yf.set_tz_cache_location(temp_cache_dir)

def get_current_z_score():
    """Fetch the latest window of daily prices and compute current Z-score."""
    raw = yf.download(tickers=[T1, T2], period="3mo", interval="1d", progress=False)['Close']
    df = raw[[T1, T2]].dropna().copy()
    
    df['spread'] = df[T1] - BETA * df[T2]
    mean = df['spread'].rolling(window=LOOKBACK).mean()
    std = df['spread'].rolling(window=LOOKBACK).std()
    df['z_score'] = (df['spread'] - mean) / std
    
    current_z = float(df['z_score'].iloc[-1])
    p1 = float(df[T1].iloc[-1])
    p2 = float(df[T2].iloc[-1])
    return current_z, p1, p2

def run_execution_cycle():
    # paper=True routes orders to simulated paper environment
    client = TradingClient(api_key=API_KEY, secret_key=SECRET_KEY, paper=True)
    
    # 1. Fetch Dynamic Account Equity
    account = client.get_account()
    total_equity = float(account.equity)
    cash_balance = float(account.cash)
    allocation_per_leg = total_equity * EQUITY_ALLOCATION_PCT

    # 2. Compute Market Metrics & Z-Score
    z, p1, p2 = get_current_z_score()
    
    # 3. Dynamic Share Sizing
    qty1 = max(1, int(allocation_per_leg / p1))
    qty2 = max(1, int((allocation_per_leg * BETA) / p2))

    print("=" * 60)
    print("                ACCOUNT & MARKET OVERVIEW                ")
    print("=" * 60)
    print(f"Total Equity:          ${total_equity:,.2f}")
    print(f"Available Cash:        ${cash_balance:,.2f}")
    print(f"Target Per-Leg Size:   ${allocation_per_leg:,.2f} ({EQUITY_ALLOCATION_PCT * 100:.0f}% of Equity)")
    print(f"Current Asset Prices:  {T1} = ${p1:.2f} | {T2} = ${p2:.2f}")
    print(f"Calculated Quantities: {qty1} shares {T1} | {qty2} shares {T2}")
    print(f"Current Spread Z-Score:{z: .4f}")
    print("=" * 60)

    # 4. Check Existing Open Positions
    open_positions = {pos.symbol: pos for pos in client.get_all_positions()}
    has_position = T1 in open_positions or T2 in open_positions

    # 5. Order Execution Logic
    if not has_position:
        if z <= -Z_ENTRY:
            print(f"SIGNAL: Long Spread triggered (Z = {z:.2f} <= -{Z_ENTRY})")
            print(f"Submitting Orders: Buy {qty1} {T1}, Short {qty2} {T2}")
            
            client.submit_order(MarketOrderRequest(
                symbol=T1, qty=qty1, side=OrderSide.BUY, time_in_force=TimeInForce.DAY
            ))
            client.submit_order(MarketOrderRequest(
                symbol=T2, qty=qty2, side=OrderSide.SELL, time_in_force=TimeInForce.DAY
            ))
            
        elif z >= Z_ENTRY:
            print(f"SIGNAL: Short Spread triggered (Z = {z:.2f} >= +{Z_ENTRY})")
            print(f"Submitting Orders: Short {qty1} {T1}, Buy {qty2} {T2}")
            
            client.submit_order(MarketOrderRequest(
                symbol=T1, qty=qty1, side=OrderSide.SELL, time_in_force=TimeInForce.DAY
            ))
            client.submit_order(MarketOrderRequest(
                symbol=T2, qty=qty2, side=OrderSide.BUY, time_in_force=TimeInForce.DAY
            ))
        else:
            print(f"Status: Neutral (Z = {z:.2f}). No entry threshold breached.")
    else:
        print("Status: Active pair position detected.")
        if abs(z) <= Z_EXIT:
            print(f"SIGNAL: Mean reversion complete (Z = {z:.2f}). Liquidating positions...")
            client.close_position(T1)
            client.close_position(T2)
        else:
            print(f"Holding open position until Z reaches ±{Z_EXIT} (Current Z = {z:.2f}).")

if __name__ == "__main__":
    run_execution_cycle()