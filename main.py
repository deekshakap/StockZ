from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import sys
import os

# Make sure the ml folder is importable
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from ml.models import (
    backtest_portfolio,
    risk_score_portfolio,
    compare_portfolios,
    find_similar_market_periods,
)

app = FastAPI(
    title="StockZ API",
    description="Backend for StockZ portfolio backtesting and risk analysis",
    version="1.0.0"
)

# Allow your React frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock this down to your frontend URL when you deploy
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────

class PortfolioRequest(BaseModel):
    tickers: List[str]
    weights: List[float]
    period: str = "3y"

    class Config:
        json_schema_extra = {
            "example": {
                "tickers": ["AAPL", "MSFT", "GOOGL"],
                "weights": [0.4, 0.4, 0.2],
                "period": "3y"
            }
        }

class CompareRequest(BaseModel):
    current_tickers: List[str]
    current_weights: List[float]
    new_tickers: List[str]
    new_weights: List[float]
    period: str = "3y"

    class Config:
        json_schema_extra = {
            "example": {
                "current_tickers": ["AAPL", "MSFT"],
                "current_weights": [0.5, 0.5],
                "new_tickers": ["NVDA", "TSLA", "AMZN"],
                "new_weights": [0.4, 0.3, 0.3],
                "period": "3y"
            }
        }

class SimilarityRequest(BaseModel):
    benchmark: str = "^IXIC"
    window: int = 60
    top_n: int = 3


# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "StockZ API is running"}


@app.post("/backtest")
def run_backtest(req: PortfolioRequest):
    """
    Simulates how a portfolio would have performed historically.
    Returns cumulative return, annualized return, volatility, and Sharpe ratio.
    """
    if round(sum(req.weights), 5) != 1.0:
        raise HTTPException(status_code=400, detail="Weights must sum to 1.0")
    if len(req.tickers) != len(req.weights):
        raise HTTPException(status_code=400, detail="Tickers and weights must be the same length")

    result = backtest_portfolio(req.tickers, req.weights, req.period)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@app.post("/risk-score")
def run_risk_score(req: PortfolioRequest):
    """
    Returns a 1-10 risk score for a portfolio based on
    volatility, max drawdown, and correlation between holdings.
    """
    if round(sum(req.weights), 5) != 1.0:
        raise HTTPException(status_code=400, detail="Weights must sum to 1.0")

    result = risk_score_portfolio(req.tickers, req.weights, req.period)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@app.post("/compare")
def run_comparison(req: CompareRequest):
    """
    Compares a current portfolio vs a new proposed one.
    Returns backtest and risk metrics side by side with a diff summary.
    """
    if round(sum(req.current_weights), 5) != 1.0:
        raise HTTPException(status_code=400, detail="Current weights must sum to 1.0")
    if round(sum(req.new_weights), 5) != 1.0:
        raise HTTPException(status_code=400, detail="New weights must sum to 1.0")

    result = compare_portfolios(
        req.current_tickers,
        req.current_weights,
        req.new_tickers,
        req.new_weights,
        req.period
    )

    return result


@app.post("/similar-periods")
def run_similarity(req: SimilarityRequest):
    """
    Uses KNN to find historical market periods that look like today,
    and shows what happened in the 30 days after each match.
    """
    result = find_similar_market_periods(
        benchmark_ticker=req.benchmark,
        window=req.window,
        top_n=req.top_n
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result