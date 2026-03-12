# robustness.py
# Vittorio Messana, 2026
#
# This is the part I think matters most.
# Any strategy looks good if you tune the parameters enough.
# The real question is whether the results hold up when I change my assumptions.
#
# I'm testing three things:
# 1. does it still work with fewer or more stocks?
# 2. does quarterly rebalancing give similar results to monthly?
# 3. does it work out of sample - i.e. on data I didn't use to build it?
#
# The in/out of sample split is the most important one.
# If the strategy only works on historical data it's useless going forward.
# I split at 2019 - anything before is in-sample, anything after is out-of-sample.
# I genuinely don't know what the out-of-sample results will be before running this.

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
returns     = returns.loc[common]
spy_returns = spy_returns.loc[common].squeeze()
regimes     = regimes.loc[common]

def run_strategy(comp, mom, lv, ret, spy, reg, n_stocks=10, freq="ME"):
    factor_map   = {"Bull": mom, "Bear": lv, "Crisis": lv}
    exposure_map = {"Bull": 1.0, "Bear": 1.0, "Crisis": 0.5}
    port, spy_r, dates = [], [], []
    ends = comp.resample(freq).last().index
    for i in range(len(ends) - 1):
        rd = ends[i]
        nd = ends[i + 1]
        if rd not in comp.index or rd not in reg.index:
            continue
        regime   = reg.loc[rd, "Regime"]
        factor   = factor_map[regime]
        exposure = exposure_map[regime]
        if rd not in factor.index:
            continue
        scores = factor.loc[rd].dropna()
        top    = scores.nlargest(n_stocks).index.tolist()
        mask   = (ret.index > rd) & (ret.index <= nd)
        period = ret.loc[mask, top]
        if period.empty:
            continue
        for date, r in (period.mean(axis=1) * exposure).items():
            port.append(r)
            dates.append(date)
            spy_r.append(spy.loc[date] if date in spy.index else np.nan)
    s   = pd.Series(port,  index=dates).dropna()
    b   = pd.Series(spy_r, index=dates).dropna()
    idx = s.index.intersection(b.index)
    return s.loc[idx], b.loc[idx]

def sharpe(r):
    return (r.mean() * 252) / (r.std() * np.sqrt(252)) if r.std() > 0 else 0

# test 1 - how many stocks
print("\ntest 1: varying number of stocks...")
results_stocks = []
for n in [5, 10, 20, 30]:
    s, b = run_strategy(composite, momentum, low_vol, returns, spy_returns, regimes, n_stocks=n)
    results_stocks.append({
        "n stocks":        n,
        "strategy sharpe": round(sharpe(s), 2),
        "spy sharpe":      round(sharpe(b), 2),
        "strategy return": f"{s.mean()*252:.2%}"
    })
    print(f"  n={n}: strategy={sharpe(s):.2f}, spy={sharpe(b):.2f}")

pd.DataFrame(results_stocks).to_csv("results/robustness_n_stocks.csv", index=False)

# test 2 - rebalancing frequency
print("\ntest 2: rebalancing frequency...")
results_freq = []
for freq, label in [("ME", "monthly"), ("QE", "quarterly")]:
    s, b = run_strategy(composite, momentum, low_vol, returns, spy_returns, regimes, freq=freq)
    results_freq.append({
        "frequency":       label,
        "strategy sharpe": round(sharpe(s), 2),
        "spy sharpe":      round(sharpe(b), 2),
        "strategy return": f"{s.mean()*252:.2%}"
    })
    print(f"  {label}: strategy={sharpe(s):.2f}, spy={sharpe(b):.2f}")

pd.DataFrame(results_freq).to_csv("results/robustness_frequency.csv", index=False)

# test 3 - in sample vs out of sample
# this is the one that really matters
# if out of sample falls apart, the strategy is probably overfit
print("\ntest 3: in-sample vs out-of-sample (split at 2019)...")

split = "2019-01-01"

for label, before_split in [("in-sample (2010-2018)", True), ("out-of-sample (2019-2025)", False)]:
    mask = composite.index < split if before_split else composite.index >= split
    s, b = run_strategy(
        composite.loc[mask], momentum.loc[mask], low_vol.loc[mask],
        returns.loc[mask],   spy_returns.loc[mask], regimes.loc[mask]
    )
    if len(s) < 50:
        continue
    print(f"\n  {label}")
    print(f"    strategy sharpe: {sharpe(s):.2f}")
    print(f"    spy sharpe:      {sharpe(b):.2f}")
    print(f"    strategy return: {s.mean()*252:.2%}")

# summary chart
print("\ngenerating robustness chart...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("robustness tests", fontsize=12)

axes[0].bar(
    [str(r["n stocks"]) for r in results_stocks],
    [r["strategy sharpe"] for r in results_stocks],
    color="#1f77b4", alpha=0.7
)
axes[0].axhline(results_stocks[0]["spy sharpe"], color="orange", linestyle="--", label="SPY")
axes[0].set_title("sharpe vs n stocks")
axes[0].set_xlabel("stocks held")
axes[0].set_ylabel("sharpe ratio")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].bar(
    [r["frequency"] for r in results_freq],
    [r["strategy sharpe"] for r in results_freq],
    color="#2ca02c", alpha=0.7
)
axes[1].set_title("sharpe vs rebalance frequency")
axes[1].set_ylabel("sharpe ratio")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/robustness_summary.png", dpi=150, bbox_inches="tight")
plt.close()

print("saved results/robustness_summary.png")
print("\nif out-of-sample sharpe is much lower than in-sample, the model is overfit")
print("I'm documenting whatever the results show, good or bad")
print("done")
