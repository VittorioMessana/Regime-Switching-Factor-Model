# Regime-Switching-Factor-Model
HMM-gated equity factor strategies with honest OOS testing.
# Regime-Switching Factor Model

A quantitative research project implementing a Hidden Markov Model (HMM) to detect market regimes (bull, bear, crisis) and rotate equity factor exposures accordingly.

## What This Project Does
- Downloads 15 years of S&P 500 stock data (2010–2025)
- Computes six equity factors: momentum, value, quality, size, low-volatility, profitability
- Uses a Hidden Markov Model to classify the market into regimes
- Builds factor strategies that adapt to each regime
- Tests everything out-of-sample with honest performance reporting

## Honest Limitations
- Regime misclassification is a real problem — the model sometimes gets it wrong
- Backtest results are not guaranteed to repeat out of sample
- Transaction costs are estimated, not exact

## Project Structure
- `src/` — all Python source code
- `data/` — raw and processed data
- `notebooks/` — step by step analysis notebooks
- - `results/` — charts and output tables

## Libraries Used
- yfinance — market data
- hmmlearn — hidden markov models
- pandas / numpy — data processing
- scikit-learn — factor construction
- matplotlib / seaborn — visualisation

## Author
Vittorio Messana, 2026
