import os
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Cache management
temp_cache_dir = os.path.join(tempfile.gettempdir(), "yfinance_cache")
os.makedirs(temp_cache_dir, exist_ok=True)
yf.set_tz_cache_location(temp_cache_dir)

def run_stat_arb_backtest(
    t1: str = "SPGI", 
    t2: str = "MCO", 
    beta: float = 1.0426, 
    start_date: str = "2024-01-01", 
    end_date: str = "2026-09-01", 
    window: int = 30, 
    z_entry: float = 2.0, 
    z_exit: float = 0.5,
    slippage_bps: float = 5.0 # 5 basis points = 0.05% friction per trade
):
    print(f"Downloading out-of-sample testing data ({start_date} to {end_date})...")
    raw = yf.download(tickers=[t1, t2], start=start_date, end=end_date, progress=False)['Close']
    df = raw[[t1, t2]].dropna().copy()

    # 1. Compute Spread using the in-sample Beta
    df['spread'] = df[t1] - beta * df[t2]

    # 2. Rolling Z-Score (strictly backward-looking to prevent lookahead bias)
    df['rolling_mean'] = df['spread'].rolling(window=window).mean()
    df['rolling_std'] = df['spread'].rolling(window=window).std()
    df['z_score'] = (df['spread'] - df['rolling_mean']) / df['rolling_std']

    # 3. Simulate Position Execution
    # Pos =  1: Long Spread (Long t1, Short t2)
    # Pos = -1: Short Spread (Short t1, Long t2)
    # Pos =  0: Neutral / Flat
    positions = [0] * len(df)
    current_pos = 0

    for i in range(len(df)):
        z = df['z_score'].iloc[i]
        if np.isnan(z):
            positions[i] = 0
            continue

        if current_pos == 0:
            if z <= -z_entry:
                current_pos = 1
            elif z >= z_entry:
                current_pos = -1
        elif current_pos == 1:
            if z >= -z_exit:
                current_pos = 0
        elif current_pos == -1:
            if z <= z_exit:
                current_pos = 0

        positions[i] = current_pos

    df['position'] = positions

    # 4. Returns & Transaction Costs
    # Shift position by 1 day because today's signal executes for tomorrow's return
    df['held_position'] = df['position'].shift(1).fillna(0)
    
    # Asset daily simple returns
    ret1 = df[t1].pct_change().fillna(0)
    ret2 = df[t2].pct_change().fillna(0)

    # Dollar-neutral spread daily return: 0.5 * ret1 - 0.5 * ret2
    spread_return = 0.5 * ret1 - 0.5 * ret2
    strategy_gross_return = df['held_position'] * spread_return

    # Deduct transaction friction on position changes
    trades = df['position'].diff().abs().fillna(0)
    friction = trades * (slippage_bps / 10000.0)
    df['net_return'] = strategy_gross_return - friction

    # Cumulative growth
    df['equity_curve'] = (1.0 + df['net_return']).cumprod()
    df['benchmark_spgi'] = (1.0 + ret1).cumprod()

    # 5. Performance Metrics
    trading_days = 252
    net_ret = df['net_return']
    annualized_return = (df['equity_curve'].iloc[-1]) ** (trading_days / len(df)) - 1.0
    annualized_vol = net_ret.std() * np.sqrt(trading_days)
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0.0

    # Drawdown
    peak = df['equity_curve'].cummax()
    drawdown = (df['equity_curve'] - peak) / peak
    max_drawdown = drawdown.min()

    print("\n" + "=" * 55)
    print("           OUT-OF-SAMPLE PERFORMANCE REPORT           ")
    print("=" * 55)
    print(f"Annualized Net Return: {annualized_return * 100:.2f}%")
    print(f"Annualized Volatility: {annualized_vol * 100:.2f}%")
    print(f"Sharpe Ratio:          {sharpe_ratio:.2f}")
    print(f"Max Drawdown:          {max_drawdown * 100:.2f}%")
    print(f"Total Trades Fired:    {int(trades.sum())}")
    print("=" * 55)

    # 6. Plotting Tearsheet
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax1.plot(df.index, df['equity_curve'], label='Stat-Arb Strategy (Net)', color='#00d084', linewidth=1.8)
    ax1.plot(df.index, df['benchmark_spgi'], label=f'Buy & Hold {t1}', color='#888888', linestyle='--', alpha=0.7)
    ax1.set_title(f'Statistical Arbitrage Engine: {t1} vs {t2} (Out-of-Sample)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Growth of $1.00')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(df.index, df['z_score'], label='Spread Z-Score', color='#007acc', linewidth=1.2)
    ax2.axhline(z_entry, color='red', linestyle='--', alpha=0.7, label=f'Short Entry (+{z_entry}σ)')
    ax2.axhline(-z_entry, color='green', linestyle='--', alpha=0.7, label=f'Long Entry (-{z_entry}σ)')
    ax2.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    ax2.set_ylabel('Z-Score')
    ax2.set_xlabel('Date')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("tearsheet.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    run_stat_arb_backtest()