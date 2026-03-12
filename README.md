# Regime-Switching Factor Model

A quantitative finance project I built to explore whether equity factor 
strategies perform differently depending on the state of the market.

## The idea

Markets cycle through different states - bull runs, bear markets, crashes.
I wanted to know whether you could detect these states algorithmically
and use that information to rotate between different factor strategies.

The answer, based on my results, is [update this once you run the code].

## How it works

1. Download 15 years of S&P 500 stock data and VIX
2. Build six equity factors: momentum, value, quality, size, low-vol, profitability
3. Train a Hidden Markov Model on SPY returns and VIX to detect market regimes
4. Run a monthly rebalancing strategy that picks different factors per regime
5. Test whether the results hold up out-of-sample and under different assumptions

## What I found

[Update this section with your actual results after running the code]

- In-sample Sharpe: 
- Out-of-sample Sharpe: 
- SPY Sharpe for comparison: 
- Most common regime detected: 

## Honest limitations

- My value and size factors are price-based proxies. Real implementations 
  use accounting data (P/E ratios, market cap) which I don't have for free.
- The HMM regime labels are sensitive to the number of states chosen.
  I tested 2 and 3 states in robustness.py - results vary.
- Backtests always look better than live trading. Transaction costs here 
  are estimated, not exact.
- Out-of-sample performance is what actually matters. In-sample results 
  are not a reliable guide to future returns.

## Files

- `src/data_collection.py` - downloads all market data
- `src/factor_construction.py` - builds the six factors
- `src/hmm_model.py` - trains the HMM and classifies regimes
- `src/strategy.py` - runs the factor strategy month by month
- `src/robustness.py` - tests whether results hold under different assumptions

## Libraries

yfinance, hmmlearn, pandas, numpy, scikit-learn, matplotlib

## Author

Vittorio Messana, 2026
