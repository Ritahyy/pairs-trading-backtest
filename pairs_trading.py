"""
Pairs Trading — Statistical Arbitrage Backtest
================================================
Generates a cointegrated pair, tests for cointegration,
trades the spread on z-score signals, and reports performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import coint, adfuller
import warnings
warnings.filterwarnings("ignore")

# ── 1. Simulate cointegrated pair ─────────────────────────────────────────────

def generate_pair(n=756, seed=42):
    np.random.seed(seed)
    dt = pd.date_range("2022-01-03", periods=n, freq="B")
    common   = np.cumsum(np.random.normal(0, 0.008, n))
    noise_a  = np.cumsum(np.random.normal(0, 0.004, n))
    noise_b  = np.cumsum(np.random.normal(0, 0.004, n))
    mr_noise = np.zeros(n)
    for i in range(1, n):
        mr_noise[i] = 0.92 * mr_noise[i-1] + np.random.normal(0, 0.015)
    log_a = 3.50 + common + noise_a
    log_b = 3.20 + 0.92 * common + noise_b + mr_noise
    return pd.DataFrame({"Asset_A": np.exp(log_a)*30, "Asset_B": np.exp(log_b)*22}, index=dt)

# ── 2. Hedge ratio via OLS ────────────────────────────────────────────────────

def ols_hedge(A, B):
    """y = alpha + beta*x  →  returns (beta, alpha)"""
    X = np.column_stack([np.ones(len(B)), B.values])
    coef = np.linalg.lstsq(X, A.values, rcond=None)[0]
    return coef[1], coef[0]   # beta, alpha

# ── 3. Cointegration tests ────────────────────────────────────────────────────

def run_tests(prices):
    A, B = prices["Asset_A"], prices["Asset_B"]
    _, pval, _ = coint(A, B)
    tag = "COINTEGRATED ✓" if pval < 0.05 else "NOT cointegrated ✗"
    print(f"Engle-Granger p-value: {pval:.4f}  ({tag})")

    beta, alpha = ols_hedge(A, B)
    spread = A - beta * B - alpha
    adf_p = adfuller(spread)[1]
    tag2 = "Stationary ✓" if adf_p < 0.05 else "Non-stationary ✗"
    print(f"ADF on spread p-value: {adf_p:.4f}  ({tag2})")
    print(f"Hedge ratio β:         {beta:.4f}")
    return spread, beta, alpha

# ── 4. Z-score & signals ──────────────────────────────────────────────────────

def compute_signals(spread, window=60, entry=2.0, exit_th=0.5):
    mu  = spread.rolling(window).mean()
    sig = spread.rolling(window).std()
    z   = (spread - mu) / sig
    pos = np.zeros(len(z))
    state = 0
    for i in range(window, len(z)):
        zi = z.iloc[i]
        if state == 0:
            if   zi >  entry: state = -1
            elif zi < -entry: state =  1
        elif state == 1  and zi >= -exit_th: state = 0
        elif state == -1 and zi <=  exit_th: state = 0
        pos[i] = state
    return z, pd.Series(pos, index=spread.index)

# ── 5. Backtest ───────────────────────────────────────────────────────────────

def backtest(prices, position, beta, alpha, tc_bps=5):
    A, B = prices["Asset_A"], prices["Asset_B"]
    spread = A - beta * B - alpha
    spread_chg = spread.diff().fillna(0)
    pnl = position.shift(1).fillna(0) * spread_chg
    trades = position.diff().abs().fillna(0)
    tc = trades * (tc_bps / 10000) * (A + beta * B) / 2
    pnl -= tc
    cum_pnl = pnl.cumsum()

    capital = (A + beta * B)
    daily_ret = pnl / capital
    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252)
    n_trades = int(trades.sum() / 2)
    peak = cum_pnl.cummax(); max_dd = (cum_pnl - peak).min()
    win_rate = (pnl[position.shift(1) != 0] > 0).mean()

    print(f"\n── Backtest Results ───────────────────────")
    print(f"  Annualised Sharpe:   {sharpe:.2f}")
    print(f"  Total PnL:           {cum_pnl.iloc[-1]:.2f}")
    print(f"  Max drawdown:        {max_dd:.2f}")
    print(f"  Round trips:         {n_trades}")
    print(f"  Win rate:            {win_rate*100:.1f}%")
    return pnl, cum_pnl

# ── 6. Plotting ───────────────────────────────────────────────────────────────

def plot_all(prices, spread, z, position, cum_pnl):
    fig, axes = plt.subplots(4, 1, figsize=(13, 14), sharex=True)
    fig.suptitle("Pairs Trading — Statistical Arbitrage Backtest",
                 fontsize=13, fontweight="bold")

    ax, ax2 = axes[0], axes[0].twinx()
    ax.plot(prices.index, prices["Asset_A"], color="#1A4F8A", lw=1.2, label="Asset A")
    ax2.plot(prices.index, prices["Asset_B"], color="#D62728", lw=1.2, label="Asset B")
    l1,n1 = ax.get_legend_handles_labels(); l2,n2 = ax2.get_legend_handles_labels()
    ax.legend(l1+l2, n1+n2, fontsize=8, loc="upper left")
    ax.set_title("Asset Prices"); ax.grid(alpha=0.3)

    axes[1].plot(spread.index, spread, color="#2CA02C", lw=1)
    axes[1].axhline(spread.mean(), color="black", ls="--", lw=0.8, label="Mean")
    axes[1].set_title("Spread  A − β·B − α"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    axes[2].plot(z.index, z, color="#9467BD", lw=1)
    for th, ls, label in [(2,  "--", "Entry ±2σ"), (-2, "--", None),
                           (0.5, ":", "Exit ±0.5σ"), (-0.5, ":", None)]:
        axes[2].axhline(th, color="#D62728" if abs(th)==2 else "#2CA02C",
                        ls=ls, lw=0.8, label=label)
    axes[2].fill_between(z.index, position, 0, alpha=0.15, color="#1A4F8A", label="Position")
    axes[2].set_title("Z-Score & Position"); axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

    axes[3].plot(cum_pnl.index, cum_pnl, color="#1A4F8A", lw=1.5)
    axes[3].fill_between(cum_pnl.index, cum_pnl, 0,
                         where=cum_pnl >= 0, alpha=0.15, color="#2CA02C")
    axes[3].fill_between(cum_pnl.index, cum_pnl, 0,
                         where=cum_pnl < 0, alpha=0.15, color="#D62728")
    axes[3].set_title("Cumulative PnL"); axes[3].grid(alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=30); plt.tight_layout()
    plt.savefig("pairs_trading_backtest.png", dpi=150, bbox_inches="tight")
    print("Saved: pairs_trading_backtest.png"); plt.close()

# ── 7. Main ───────────────────────────────────────────────────────────────────

def main():
    print("Generating simulated cointegrated pair...")
    prices = generate_pair()
    print("── Cointegration Tests ─────────────────────")
    spread, beta, alpha = run_tests(prices)
    z, position = compute_signals(spread)
    pnl, cum_pnl = backtest(prices, position, beta, alpha)
    plot_all(prices, spread, z, position, cum_pnl)

if __name__ == "__main__":
    main()
