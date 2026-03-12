# strategy.py
# Vittorio Messana, 2026
#
# Now I combine the factors and regimes into an actual strategy.
# The logic is straightforward:
# different factors work better in different market conditions.
# In a bull market momentum is king.
# In a bear or crisis I want defensive low-volatility stocks.
#
# I rebalance monthly - daily would rack up too many transaction costs
# and quarterly felt too slow to react to regime changes.
#
# In a crisis I also cut exposure to 50% - the other 50% sits in cash.
# This is conservative but I'd rather preserve capital than chase returns
# when the model thinks everything is on fire.

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

print("loading data...")

composite   = pd.read_csv("data/composite_factor.csv",    index_col=0, parse_dates=True)
momentum    = pd.read_csv("data/factor_momentum.csv",     index_col=0, parse_dates=True)
low_vol     = pd.read_csv("data/factor_lowvol.csv",       index_col=0, parse_dates=True)
quality     = pd.read_csv("data/factor_quality.csv",      index_col=0, parse_dates=True)
returns     = pd.read_csv("data/returns_cut.csv",         index_col=0, parse_dates=True)
spy_returns = pd.read_csv("data/spy_returns.csv",         index_col=0, parse_dates=True)
regimes     = pd.read_csv("data/regime_labels.csv",       index_col=0, parse_dates=True)

regimes.columns = ["Regime"]

common = composite.index\
    .intersection(returns.index)\
    .intersection(regimes.index)\
    .intersection(spy_returns.index)

composite   = composite.loc[common]
momentum    = momentum.loc[common]
low_vol     = low_vol.loc[common]
quality     = quality.loc[common]
returns     = returns.loc[common]
spy_returns = spy_returns.loc[common].squeeze()
regimes     = regimes.loc[common]

print(f"{len(common)} days")

# which factor to use in each regime
FACTOR_MAP = {
    "Bull":   momentum,
    "Bear":   low_vol,
    "Crisis": low_vol
}

# how much of the portfolio to actually deploy
# pulling back to 50% in crisis - rest is cash
EXPOSURE_MAP = {
    "Bull":   1.0,
    "Bear":   1.0,
    "Crisis": 0.5
}

N_STOCKS = 10

print("running monthly rebalance...")

portfolio_returns = []
portfolio_dates   = []
spy_ret_list      = []
holdings_log      = []

month_ends = composite.resample("ME").last().index

for i in range(len(month_ends) - 1):
    rebalance_date = month_ends[i]
    next_rebalance = month_ends[i + 1]

    if rebalance_date not in composite.index:
        continue

    regime   = regimes.loc[rebalance_date, "Regime"]
    factor   = FACTOR_MAP[regime]
    exposure = EXPOSURE_MAP[regime]

    if rebalance_date not in factor.index:
        continue

    # picking top N stocks by factor score on rebalance date
    factor_scores = factor.loc[rebalance_date].dropna()
    top_stocks    = factor_scores.nlargest(N_STOCKS).index.tolist()

    # getting returns for the next month
    mask           = (returns.index > rebalance_date) & (returns.index <= next_rebalance)
    period_returns = returns.loc[mask, top_stocks]

    if period_returns.empty:
        continue

    # equal weight - all 10 stocks get the same allocation
    daily_ret = period_returns.mean(axis=1) * exposure

    for date, ret in daily_ret.items():
        portfolio_returns.append(ret)
        portfolio_dates.append(date)
        spy_ret_list.append(spy_returns.loc[date] if date in spy_returns.index else np.nan)

    holdings_log.append({
        "date":     rebalance_date,
        "regime":   regime,
        "stocks":   ", ".join(top_stocks),
        "exposure": exposure
    })

# building return series
strategy_series = pd.Series(portfolio_returns, index=portfolio_dates).sort_index()
spy_series      = pd.Series(spy_ret_list,      index=portfolio_dates).sort_index()

strategy_series = strategy_series.dropna()
spy_series      = spy_series.dropna()
common_idx      = strategy_series.index.intersection(spy_series.index)
strategy_series = strategy_series.loc[common_idx]
spy_series      = spy_series.loc[common_idx]

# performance metrics
def compute_metrics(r, label):
    ann_ret = r.mean() * 252
    ann_vol = r.std()  * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0
    cum     = (1 + r).cumprod()
    max_dd  = ((cum - cum.cummax()) / cum.cummax()).min()
    return {
        "label":           label,
        "ann. return":     f"{ann_ret:.2%}",
        "ann. volatility": f"{ann_vol:.2%}",
        "sharpe":          f"{sharpe:.2f}",
        "max drawdown":    f"{max_dd:.2%}",
        "days":            len(r)
    }

strat_m = compute_metrics(strategy_series, "regime strategy")
spy_m   = compute_metrics(spy_series,      "SPY")

print("\nresults:")
for k, v in strat_m.items():
    print(f"  {k}: {v}")
print()
for k, v in spy_m.items():
    print(f"  {k}: {v}")

# chart
cum_strat = (1 + strategy_series).cumprod()
cum_spy   = (1 + spy_series).cumprod()

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(cum_strat.index, cum_strat.values, label="regime strategy", linewidth=1.5)
ax.plot(cum_spy.index,   cum_spy.values,   label="SPY",             linewidth=1.5)
ax.set_title("regime strategy vs SPY")
ax.set_ylabel("cumulative return")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/strategy_performance.png", dpi=150, bbox_inches="tight")
plt.close()

# saving
pd.DataFrame(holdings_log).to_csv("results/holdings_log.csv", index=False)
pd.DataFrame([strat_m, spy_m]).to_csv("results/performance_metrics.csv", index=False)

print("\nsaved results")
print("done")
