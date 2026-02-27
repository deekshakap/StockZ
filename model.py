import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────
# 1. DATA FETCHING
# ─────────────────────────────────────────

def fetch_stock_data(tickers: list, period: str = "5y") -> dict:
    """
    Fetch historical closing price data for a list of tickers.
    period options: 1y, 2y, 5y, 10y, max
    """
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, period=period, progress=False)
            if not df.empty:
                data[ticker] = df["Close"]
        except Exception as e:
            print(f"Could not fetch {ticker}: {e}")
    return data


# ─────────────────────────────────────────
# 2. BACKTESTING
# ─────────────────────────────────────────

def backtest_portfolio(tickers: list, weights: list, period: str = "5y") -> dict:
    """
    Simulates how a portfolio would have performed historically.
    
    tickers: list of stock symbols e.g. ["AAPL", "MSFT", "GOOGL"]
    weights: list of floats that sum to 1 e.g. [0.4, 0.4, 0.2]
    period: how far back to test
    
    Returns cumulative return, annualized return, and daily portfolio value series.
    """
    assert len(tickers) == len(weights), "Tickers and weights must match in length"
    assert round(sum(weights), 5) == 1.0, "Weights must sum to 1"

    raw = fetch_stock_data(tickers, period)

    # Align all tickers on same date index
    prices = pd.DataFrame(raw).dropna()

    if prices.empty:
        return {"error": "No price data returned. Check your tickers."}

    # Normalize to starting price = 1
    normalized = prices / prices.iloc[0]

    # Weighted portfolio value
    portfolio = (normalized * weights).sum(axis=1)

    # Daily returns
    daily_returns = portfolio.pct_change().dropna()

    # Metrics
    cumulative_return = (portfolio.iloc[-1] - 1) * 100
    trading_days = len(daily_returns)
    years = trading_days / 252
    annualized_return = ((portfolio.iloc[-1]) ** (1 / years) - 1) * 100 if years > 0 else 0
    volatility = daily_returns.std() * np.sqrt(252) * 100
    sharpe = (annualized_return / volatility) if volatility != 0 else 0

    return {
        "tickers": tickers,
        "weights": weights,
        "cumulative_return_pct": round(cumulative_return, 2),
        "annualized_return_pct": round(annualized_return, 2),
        "volatility_pct": round(volatility, 2),
        "sharpe_ratio": round(sharpe, 2),
        "portfolio_series": portfolio.to_dict(),  # date -> value, for charting
        "start_date": str(prices.index[0].date()),
        "end_date": str(prices.index[-1].date()),
    }


# ─────────────────────────────────────────
# 3. RISK SCORING
# ─────────────────────────────────────────

def risk_score_portfolio(tickers: list, weights: list, period: str = "2y") -> dict:
    """
    Analyzes portfolio risk based on:
    - Volatility
    - Max drawdown
    - Correlation between holdings
    Returns a 1-10 risk score (10 = highest risk)
    """
    raw = fetch_stock_data(tickers, period)
    prices = pd.DataFrame(raw).dropna()

    if prices.empty:
        return {"error": "No price data returned."}

    daily_returns = prices.pct_change().dropna()

    # Weighted portfolio returns
    portfolio_returns = (daily_returns * weights).sum(axis=1)

    # Volatility (annualized)
    volatility = portfolio_returns.std() * np.sqrt(252)

    # Max drawdown
    cumulative = (1 + portfolio_returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Average pairwise correlation
    corr_matrix = daily_returns.corr()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    avg_correlation = upper.stack().mean() if not upper.stack().empty else 0

    # Risk score: scale volatility 0-10 (annualized vol > 40% = very high risk)
    vol_score = min(volatility / 0.40 * 10, 10)
    drawdown_score = min(abs(max_drawdown) / 0.50 * 10, 10)
    correlation_score = avg_correlation * 10  # high correlation = less diversification

    risk_score = round((vol_score * 0.5 + drawdown_score * 0.3 + correlation_score * 0.2), 1)
    risk_score = min(max(risk_score, 0), 10)

    if risk_score <= 3:
        label = "Low"
    elif risk_score <= 6:
        label = "Medium"
    elif risk_score <= 8:
        label = "High"
    else:
        label = "Very High"

    return {
        "risk_score": risk_score,
        "risk_label": label,
        "annualized_volatility_pct": round(volatility * 100, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "avg_correlation": round(avg_correlation, 3),
    }


# ─────────────────────────────────────────
# 4. HISTORICAL SIMILARITY MATCHING (KNN)
# ─────────────────────────────────────────

def find_similar_market_periods(benchmark_ticker: str = "^IXIC", window: int = 60, top_n: int = 3) -> dict:
    """
    Looks at the last `window` trading days of market behavior
    and finds the most similar historical periods using KNN.
    Then shows what happened in the 30 days AFTER each match.
    
    benchmark_ticker: use ^IXIC for NASDAQ, ^GSPC for S&P 500
    window: how many days to use as the "fingerprint"
    top_n: how many similar periods to return
    """
    df = yf.download(benchmark_ticker, period="10y", progress=False)["Close"]

    if df.empty:
        return {"error": "Could not fetch benchmark data."}

    returns = df.pct_change().dropna()

    # Build rolling windows of `window` days as feature vectors
    features = []
    dates = []
    for i in range(window, len(returns) - 30):
        window_data = returns.iloc[i - window:i].values
        features.append(window_data)
        dates.append(returns.index[i])

    features = np.array(features)

    # Scale
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Current window = last `window` days
    current_window = returns.iloc[-window:].values.reshape(1, -1)
    current_scaled = scaler.transform(current_window)

    # Fit KNN
    knn = NearestNeighbors(n_neighbors=top_n + 1, metric="euclidean")
    knn.fit(features_scaled)
    distances, indices = knn.kneighbors(current_scaled)

    results = []
    for i, idx in enumerate(indices[0][1:]):  # skip index 0 (itself if present)
        match_date = dates[idx]
        # What happened 30 days after this match?
        future_start = returns.index.get_loc(match_date)
        future_returns = returns.iloc[future_start: future_start + 30]
        cumulative_after = ((1 + future_returns).cumprod().iloc[-1] - 1) * 100

        results.append({
            "similar_period_start": str(match_date.date()),
            "similarity_distance": round(float(distances[0][i + 1]), 4),
            "what_happened_30d_after_pct": round(float(cumulative_after), 2),
        })

    return {
        "benchmark": benchmark_ticker,
        "current_window_days": window,
        "similar_historical_periods": results,
        "note": "Past patterns are not guarantees of future performance. Educational use only."
    }


# ─────────────────────────────────────────
# 5. PORTFOLIO COMPARISON
# ─────────────────────────────────────────

def compare_portfolios(
    current_tickers: list,
    current_weights: list,
    new_tickers: list,
    new_weights: list,
    period: str = "3y"
) -> dict:
    """
    Compares a user's current portfolio vs a new proposed one.
    Returns backtest and risk metrics side by side.
    """
    current_backtest = backtest_portfolio(current_tickers, current_weights, period)
    new_backtest = backtest_portfolio(new_tickers, new_weights, period)

    current_risk = risk_score_portfolio(current_tickers, current_weights)
    new_risk = risk_score_portfolio(new_tickers, new_weights)

    return {
        "current_portfolio": {**current_backtest, **current_risk},
        "new_portfolio": {**new_backtest, **new_risk},
        "comparison": {
            "return_difference_pct": round(
                new_backtest.get("cumulative_return_pct", 0) - current_backtest.get("cumulative_return_pct", 0), 2
            ),
            "risk_score_difference": round(
                new_risk.get("risk_score", 0) - current_risk.get("risk_score", 0), 1
            ),
            "sharpe_difference": round(
                new_backtest.get("sharpe_ratio", 0) - current_backtest.get("sharpe_ratio", 0), 2
            ),
        }
    }


# ─────────────────────────────────────────
# 6. QUICK TEST — run this file directly to verify everything works
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("\n--- BACKTEST ---")
    result = backtest_portfolio(["AAPL", "MSFT", "GOOGL"], [0.4, 0.4, 0.2])
    print({k: v for k, v in result.items() if k != "portfolio_series"})

    print("\n--- RISK SCORE ---")
    print(risk_score_portfolio(["AAPL", "MSFT", "GOOGL"], [0.4, 0.4, 0.2]))

    print("\n--- SIMILAR MARKET PERIODS ---")
    print(find_similar_market_periods())

    print("\n--- PORTFOLIO COMPARISON ---")
    comparison = compare_portfolios(
        ["AAPL", "MSFT"], [0.5, 0.5],
        ["NVDA", "TSLA", "AMZN"], [0.4, 0.3, 0.3]
    )
    print(comparison["comparison"])