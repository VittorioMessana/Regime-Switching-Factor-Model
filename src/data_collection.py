# data_collection.py
# Vittorio Messana, 2026
#
# First thing I need to do is get the actual data.
# I'm using yfinance because it's free and connects straight to Yahoo Finance.
# Took me a while to figure out the right syntax but this works.

import yfinance as yf
import pandas as pd
import numpy as np
import os

# dates I'm working with
START_DATE = "2010-01-01"
END_DATE   = "2025-12-31"

# I picked 50 stocks across different sectors
# wanted enough diversity without making the download take forever
TICKERS = [
    # tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSM", "AVGO", "ORCL", "ASML",
    # finance
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "AXP", "SPGI", "MCO", "ICE",
    # healthcare
    "JNJ", "UNH", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY",
    # consumer
    "PG", "KO", "PEP", "WMT", "COST", "MCD", "NKE", "SBUX", "TGT", "HD",
    # energy and industrial
    "XOM", "CVX", "CAT", "DE", "HON", "UPS", "BA", "MMM", "GE", "LMT"
]

# downloading prices
# I'm using adjusted close so dividends and stock splits don't mess up the numbers
print("downloading stock prices...")

prices = yf.download(
    tickers=TICKERS,
    start=START_DATE,
    end=END_DATE,
    auto_adjust=True
)["Close"]

print(f"got {prices.shape[0]} days for {prices.shape[1]} stocks")

# VIX is the market fear index - goes up when things get scary
# I need this for the HMM later to help it identify crisis periods
# VIX above 30 is generally considered a bad sign
print("downloading VIX...")

vix = yf.download(
    tickers="^VIX",
    start=START_DATE,
    end=END_DATE,
    auto_adjust=True
)["Close"]

vix.name = "VIX"

# SPY is just the S&P 500 ETF - I'm using it as my benchmark
# everything gets compared against this
print("downloading SPY...")

spy = yf.download(
    tickers="SPY",
    start=START_DATE,
    end=END_DATE,
    auto_adjust=True
)["Close"]

spy.name = "SPY"

# computing daily returns
# pct_change just does (today - yesterday) / yesterday
# dropna removes the first row which has no previous day to compare against
print("computing returns...")

returns     = prices.pct_change().dropna()
spy_returns = spy.pct_change().dropna()

# cleaning up
# some stocks have missing data - I'm removing anything with more than 20% gaps
# filling the rest with 0 (no return that day)
missing_pct  = returns.isnull().sum() / len(returns)
good_tickers = missing_pct[missing_pct < 0.20].index
returns      = returns[good_tickers].fillna(0)

print(f"kept {len(good_tickers)} stocks after cleaning")

# making sure all three datasets have the same dates
# otherwise things break when I try to combine them later
common_dates = returns.index.intersection(vix.index).intersection(spy_returns.index)
returns      = returns.loc[common_dates]
vix_aligned  = vix.loc[common_dates]
spy_aligned  = spy_returns.loc[common_dates]

print(f"{len(common_dates)} trading days total")
print(f"from {common_dates[0].date()} to {common_dates[-1].date()}")

# saving everything
# I'll load these csv files in the other scripts
os.makedirs("data", exist_ok=True)

returns.to_csv("data/stock_returns.csv")
vix_aligned.to_csv("data/vix.csv")
spy_aligned.to_csv("data/spy_returns.csv")
prices.loc[common_dates].to_csv("data/stock_prices.csv")

print("saved all data files")
print(f"avg VIX: {vix_aligned.mean().iloc[0]:.1f}")
print(f"max VIX: {vix_aligned.max().iloc[0]:.1f}")
print(f"SPY annualised return: {spy_aligned.mean().iloc[0] * 252:.2%}")
print("done")
