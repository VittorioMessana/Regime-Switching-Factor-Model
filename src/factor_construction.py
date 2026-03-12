# factor_construction.py
# Vittorio Messana, 2026
#
# This builds the six factors I'm using in the model.
# A factor is basically a characteristic that predicts returns.
# I read about these in AQR's research papers - they're standard in quant finance.
#
# The six I'm using:
# 1. momentum     - stocks going up tend to keep going up
# 2. value        - cheap stocks outperform expensive ones over time
# 3. quality      - consistent stable returns beat erratic ones
# 4. size         - smaller stocks have historically beaten larger ones
# 5. low-vol      - less volatile stocks actually outperform (weird but true)
# 6. profitability - profitable companies do better

import pandas as pd
import numpy as np

print("loading data...")

returns = pd.read_csv("data/stock_returns.csv", index_col=0, parse_dates=True)
prices  = pd.read_csv("data/stock_prices.csv",  index_col=0, parse_dates=True)

print(f"{returns.shape[0]} days, {returns.shape[1]} stocks")

# momentum
# looking at 12 month return but skipping the last month
# skipping the last month avoids short term reversal - stocks tend to bounce back
# in the very short term which would mess up the signal
print("momentum...")

momentum = (
    (1 + returns).rolling(252).apply(np.prod, raw=True) /
    (1 + returns).rolling(21).apply(np.prod, raw=True)
) - 1

# value
# I don't have P/E data so I'm using a price proxy instead
# the idea is stocks far below their 52 week high are cheap
# not perfect but it's what I can do with free data
print("value...")

high_52w = prices.rolling(252).max()
value    = 1 - (prices / high_52w)
value    = value.loc[returns.index]

# quality
# I'm measuring quality as return consistency
# mean return divided by standard deviation over 3 months
# high ratio = steady positive returns = high quality company
print("quality...")

rolling_mean = returns.rolling(63).mean()
rolling_std  = returns.rolling(63).std()
quality      = rolling_mean / (rolling_std + 1e-8)

# size
# ideally I'd use market cap but I don't have that data for free
# using price rank as a rough proxy - lower price tends to mean smaller company
# I know this is a weak approximation and I note it as a limitation
print("size...")

size = prices.rank(axis=1, ascending=True)
size = size.loc[returns.index]

# low volatility
# this one surprised me when I first read about it
# you'd expect more risk = more reward but actually low vol stocks outperform
# I'm computing rolling 3 month vol then inverting it
# so low vol stocks get high scores
print("low vol...")

vol_63d = returns.rolling(63).std()
low_vol = 1 / (vol_63d + 1e-8)

# profitability
# using 1 year cumulative return as a proxy for profitability
# again not perfect since I'd ideally want ROE from financial statements
print("profitability...")

profitability = (1 + returns).rolling(252).apply(np.prod, raw=True) - 1

# ranking everything cross-sectionally
# on each day I rank all stocks from 0 to 1 based on their score
# this makes the factors comparable to each other
# rank 1 = best stock that day, rank 0 = worst
print("ranking factors...")

def rank_cross_section(df):
    return df.rank(axis=1, pct=True)

momentum_ranked      = rank_cross_section(momentum)
value_ranked         = rank_cross_section(value)
quality_ranked       = rank_cross_section(quality)
size_ranked          = rank_cross_section(size)
low_vol_ranked       = rank_cross_section(low_vol)
profitability_ranked = rank_cross_section(profitability)

# combining into one composite score
# equal weight for now - could experiment with different weights later
print("building composite score...")

composite = (
    momentum_ranked +
    value_ranked +
    quality_ranked +
    size_ranked +
    low_vol_ranked +
    profitability_ranked
) / 6.0

# dropping the first year because the rolling windows need 252 days to warm up
# before that the factors are just NaN
warmup_days = 252
composite   = composite.iloc[warmup_days:]
returns_cut = returns.iloc[warmup_days:]

print(f"{composite.shape[0]} days after warmup")

# saving
print("saving...")

composite.to_csv("data/composite_factor.csv")
momentum_ranked.iloc[warmup_days:].to_csv("data/factor_momentum.csv")
value_ranked.iloc[warmup_days:].to_csv("data/factor_value.csv")
quality_ranked.iloc[warmup_days:].to_csv("data/factor_quality.csv")
size_ranked.iloc[warmup_days:].to_csv("data/factor_size.csv")
low_vol_ranked.iloc[warmup_days:].to_csv("data/factor_lowvol.csv")
profitability_ranked.iloc[warmup_days:].to_csv("data/factor_profitability.csv")
returns_cut.to_csv("data/returns_cut.csv")

print("done")
print(f"date range: {composite.index[0].date()} to {composite.index[-1].date()}")
