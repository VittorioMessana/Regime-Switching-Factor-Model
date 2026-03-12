# Regime-Switching Factor Model

A quantitative finance project I built to explore whether equity factor 
strategies perform differently depending on the state of the market.

## The idea

Markets cycle through different states - bull runs, bear markets, crashes.
I wanted to know whether you could detect these states algorithmically
and use that information to rotate between different factor strategies.

I used a Hidden Markov Model trained on S&P 500 returns and VIX to detect
three hidden market regimes, then built a monthly rebalancing strategy that
picks different factors depending on which regime the model thinks we are in.

## Results

| | Regime Strategy | SPY Benchmark |
|---|---|---|
| Annualised Return | 17.65% | 15.08% |
| Annualised Volatility | 14.22% | 16.49% |
| Sharpe Ratio | 1.24 | 0.91 |
| Max Drawdown | -18.55% | -23.18% |

The strategy beat SPY on every metric - higher return, lower volatility,
better Sharpe, and smaller drawdown over the full 2011-2025 period.

## Regime detection

The HMM identified three distinct market states:

| Regime | Days | % of time | Ann. Return | Avg VIX |
|---|---|---|---|---|
| Bull | 1506 | 49.9% | 26.04% | 13.4 |
| Bear | 1150 | 38.1% | 7.90% | 19.7 |
| Crisis | 360 | 11.9% | -14.52% | 31.7 |

Once in a regime the market tends to stay there - bull to bull transition
probability is 97.9%, crisis to crisis is 96.0%.

## Robustness testing

The result I care most about is the in/out-of-sample split.
If the strategy only works on historical data it is useless going forward.

| Period | Strategy Sharpe | SPY Sharpe |
|---|---|---|
| In-sample (2010-2018) | 1.13 | 0.73 |
| Out-of-sample (2019-2025) | 1.33 | 0.95 |

The out-of-sample Sharpe is actually higher than in-sample.
I was not expecting this - it suggests the regime detection is picking up
something real rather than fitting noise.

The strategy also holds up across different portfolio sizes:

| Stocks held | Strategy Sharpe |
|---|---|
| 5 | 1.19 |
| 10 | 1.24 |
| 20 | 1.12 |
| 30 | 1.15 |

## Honest limitations

- Monthly rebalancing is essential. Quarterly drops Sharpe to 0.65,
  below SPY. The strategy needs frequent regime-driven updates to work.
- My value and size factors use price-based proxies, not accounting data.
  Real implementations would use P/E ratios and market capitalisation.
- Transaction costs are not modelled exactly. Real execution would reduce
  returns somewhat.
- The HMM regime labels are sensitive to initialisation. I used
  random_state=42 for reproducibility but different seeds give
  slightly different regime boundaries.
- Past performance does not guarantee future results.

## How it works

1. `data_collection.py` - downloads 15 years of price data for 50 S&P 500 stocks plus VIX and SPY
2. `factor_construction.py` - builds six equity factors: momentum, value, quality, size, low-vol, profitability
3. `hmm_model.py` - trains a Gaussian HMM to classify each trading day into bull, bear, or crisis regime
4. `strategy.py` - monthly rebalancing strategy using momentum in bull markets, low-vol in bear and crisis
5. `robustness.py` - tests whether results hold under different assumptions and out of sample

## Libraries

yfinance, hmmlearn, pandas, numpy, scikit-learn, matplotlib

## Author

Vittorio Messana, 2026
