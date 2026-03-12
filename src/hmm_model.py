# hmm_model.py
# Vittorio Messana, 2026
#
# This is the core of the whole project.
# The idea is that the market has hidden states - bull, bear, crisis -
# that you can't directly observe but can infer from the data.
# A Hidden Markov Model does exactly that.
#
# I first read about HMMs in the context of speech recognition
# then found papers applying them to equity markets.
# The basic intuition: if VIX is high and returns are negative,
# we're probably in a crisis regime. The HMM formalises that logic.
#
# I'm using three states. Could have used two or four but three felt right
# after reading a few papers. I test this assumption in robustness.py.

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
import os

os.makedirs("results", exist_ok=True)

print("loading data...")

spy_returns = pd.read_csv("data/spy_returns.csv", index_col=0, parse_dates=True)
vix         = pd.read_csv("data/vix.csv",         index_col=0, parse_dates=True)

common      = spy_returns.index.intersection(vix.index)
spy_returns = spy_returns.loc[common]
vix         = vix.loc[common]

# dropping the warmup period to match factor construction
spy_returns = spy_returns.iloc[252:]
vix         = vix.iloc[252:]

print(f"{len(spy_returns)} days loaded")

# building the observation matrix
# each row is one trading day
# I'm using three signals: SPY return, VIX level, realised vol
# these three together should give the HMM enough to work with
spy_ret_values = spy_returns.values.flatten()
vix_values     = vix.values.flatten()
realized_vol   = pd.Series(spy_ret_values).rolling(21).std().fillna(0).values

X = np.column_stack([
    spy_ret_values,
    vix_values / 100,  # scaling VIX down so it's in a similar range to returns
    realized_vol
])

print(f"observation matrix: {X.shape}")

# fitting the model
# n_components=3 means three hidden states
# covariance_type=full means each state gets its own covariance matrix
# random_state=42 just makes results reproducible
print("fitting HMM - this takes a minute...")

model = GaussianHMM(
    n_components=3,
    covariance_type="full",
    n_iter=1000,
    random_state=42
)

model.fit(X)

print(f"converged: {model.monitor_.converged}")

# predicting regimes
regimes = model.predict(X)
dates   = spy_returns.index

# figuring out which state number is which regime
# I do this by looking at average VIX in each state
# highest VIX = crisis, lowest VIX = bull - makes intuitive sense
regime_vix = {}
for r in np.unique(regimes):
    mask          = regimes == r
    regime_vix[r] = vix_values[mask].mean()
    print(f"state {r}: avg VIX={regime_vix[r]:.1f}, avg return={spy_ret_values[mask].mean():.4f}")

sorted_states = sorted(regime_vix, key=regime_vix.get)
bull_state    = sorted_states[0]
bear_state    = sorted_states[1]
crisis_state  = sorted_states[2]

print(f"bull={bull_state}, bear={bear_state}, crisis={crisis_state}")

label_map     = {bull_state: "Bull", bear_state: "Bear", crisis_state: "Crisis"}
regime_labels = pd.Series([label_map[r] for r in regimes], index=dates, name="Regime")

# stats per regime
print("\nregime breakdown:")
for label in ["Bull", "Bear", "Crisis"]:
    mask    = regime_labels == label
    n       = mask.sum()
    ret     = spy_ret_values[mask.values].mean() * 252
    avg_vix = vix_values[mask.values].mean()
    pct     = n / len(regime_labels) * 100
    print(f"  {label}: {n} days ({pct:.1f}%) | ann.return={ret:.2%} | avg VIX={avg_vix:.1f}")

# chart showing regimes over time
# top panel: SPY cumulative return
# bottom panel: what regime we were in
print("\nmaking chart...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig.suptitle("S&P 500 regime detection - Hidden Markov Model", fontsize=13)

cum_ret = (1 + pd.Series(spy_ret_values, index=dates)).cumprod()
ax1.plot(dates, cum_ret, color="#1f77b4", linewidth=1)
ax1.set_ylabel("cumulative return")
ax1.grid(True, alpha=0.3)

colors     = {"Bull": "#90EE90", "Bear": "#FFD700", "Crisis": "#FF6B6B"}
prev       = regime_labels.iloc[0]
start      = dates[0]

for i, (date, regime) in enumerate(regime_labels.items()):
    if regime != prev or i == len(regime_labels) - 1:
        ax1.axvspan(start, date, alpha=0.2, color=colors[prev], label=prev)
        ax2.axvspan(start, date, alpha=0.4, color=colors[prev])
        start = date
        prev  = regime

regime_numeric = regime_labels.map({"Bull": 1, "Bear": 0, "Crisis": -1})
ax2.plot(dates, regime_numeric, color="#333333", linewidth=0.5)
ax2.set_yticks([-1, 0, 1])
ax2.set_yticklabels(["Crisis", "Bear", "Bull"])
ax2.set_ylabel("regime")
ax2.grid(True, alpha=0.3)

handles, labels = ax1.get_legend_handles_labels()
by_label        = dict(zip(labels, handles))
ax1.legend(by_label.values(), by_label.keys(), loc="upper left")

plt.tight_layout()
plt.savefig("results/regime_chart.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved results/regime_chart.png")

# saving regime labels - I'll use these in strategy.py
regime_labels.to_csv("data/regime_labels.csv")
print("saved data/regime_labels.csv")

# transition matrix
# shows probability of moving from one regime to another day to day
# interesting to see how sticky each regime is
print("\ntransition matrix:")
trans = pd.DataFrame(
    model.transmat_,
    index=[label_map[i] for i in range(3)],
    columns=[label_map[i] for i in range(3)]
)
print(trans.round(3))
trans.to_csv("results/transition_matrix.csv")
print("done")
