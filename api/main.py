from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from textblob import TextBlob


def build_model(name: str):
    """Returns (estimator, display_name). Centralizes model config so backtest and live prediction stay in sync."""
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=200, random_state=42, oob_score=True,
            max_depth=6, min_samples_leaf=5, max_features="sqrt"
        ), "Random Forest"
    if name == "lgb":
        return LGBMRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            random_state=42, n_jobs=1, verbose=-1
        ), "LightGBM"
    return LinearRegression(), "Linear Regression"


def walk_forward_directional_accuracy(train_data, features, target_col, model_name,
                                      n_windows=4, test_size=80):
    """
    Honest out-of-sample directional accuracy.
    For each window: train on everything before, predict the next test_size days,
    score whether the predicted sign matched the actual sign. Returns (accuracy, n_predictions).
    """
    correct = 0
    total = 0
    n = len(train_data)

    for i in range(n_windows):
        end_idx = n - i * test_size
        start_idx = end_idx - test_size
        if start_idx < 250:  # need a meaningful training history
            break

        train = train_data.iloc[:start_idx]
        test = train_data.iloc[start_idx:end_idx]

        m, _ = build_model(model_name)
        m.fit(train[features], train[target_col])
        preds = m.predict(test[features])
        actuals = test[target_col].values

        # Sign agreement (treat 0 as "up" — rare in pct change data anyway)
        correct += int(((preds >= 0) == (actuals >= 0)).sum())
        total += len(preds)

    if total == 0:
        return None, 0
    return round(correct / total * 100, 1), total

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/predict")
def get_prediction(model: str = "linear", horizon: int = 1, symbol: str = "SPY"):
    print(f"API called: Running {model} engine for {symbol} over {horizon} day(s)...")

    # Download target symbol and VIX
    stock = yf.download(symbol, period='max', progress=False, auto_adjust=True).tail(2500)
    vix = yf.download('^VIX', period='max', progress=False, auto_adjust=True).tail(2500)

    # Clean up MultiIndex columns
    if isinstance(stock.columns, pd.MultiIndex):
        stock.columns = stock.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    stock['VIX_Close'] = vix['Close']
    stock['VIX_Change'] = vix['Close'].pct_change() * 100

    # --- FEATURE ENGINEERING ---
    stock['SMA_50'] = stock['Close'].rolling(window=50).mean()
    stock['Today_Pct_Change'] = stock['Close'].pct_change() * 100
    stock['Target_Future_Pct'] = stock['Close'].pct_change(periods=horizon).shift(-horizon) * 100
    stock['Vol_Change'] = stock['Volume'].pct_change() * 100
    stock['Daily_Range'] = (stock['High'] - stock['Low']) / stock['Close'] * 100
    stock['SMA_200'] = stock['Close'].rolling(window=200).mean()
    stock['Dist_From_200'] = (stock['Close'] - stock['SMA_200']) / stock['SMA_200'] * 100

    delta = stock['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    stock['RSI'] = 100 - (100 / (1 + rs))

    ema_12 = stock['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = stock['Close'].ewm(span=26, adjust=False).mean()
    stock['MACD'] = ema_12 - ema_26

    stock['BB_Middle'] = stock['Close'].rolling(window=20).mean()
    stock['BB_Std'] = stock['Close'].rolling(window=20).std()
    stock['BB_Upper'] = stock['BB_Middle'] + (stock['BB_Std'] * 2)
    stock['Dist_BB_Upper'] = (stock['Close'] - stock['BB_Upper']) / stock['BB_Upper'] * 100

    stock['Day_Of_Week'] = stock.index.dayofweek
    stock['Lag_1'] = stock['Today_Pct_Change'].shift(1)
    stock['Lag_2'] = stock['Today_Pct_Change'].shift(2)
    stock['Prev_Close'] = stock['Close'].shift(1)
    stock['Gap_Pct'] = (stock['Open'] - stock['Prev_Close']) / stock['Prev_Close'] * 100

    # --- NEWS + SENTIMENT ---
    headlines = []
    news_sentiment = 0.0  # default neutral if fetch fails

    try:
        ticker_obj = yf.Ticker(symbol)
        news_items = ticker_obj.news[:5]  # grab top 5 articles

        scores = []
        for item in news_items:
            # yfinance news structure: item['content']['title']
            title = ""
            if isinstance(item.get('content'), dict):
                title = item['content'].get('title', '')
            elif isinstance(item.get('title'), str):
                title = item['title']

            if title:
                sentiment = TextBlob(title).sentiment.polarity  # -1 to +1
                scores.append(sentiment)
                link = ""
                if isinstance(item.get('content'), dict):
                    # nested content structure
                    canonical = item['content'].get('canonicalUrl', {})
                    link = canonical.get('url', '') if isinstance(canonical, dict) else ''
                elif isinstance(item.get('link'), str):
                    link = item['link']

                headlines.append({
                    "title": title,
                    "sentiment": round(sentiment, 3),
                    "url": link
                })

        if scores:
            news_sentiment = round(sum(scores) / len(scores), 4)

    except Exception as e:
        print(f"News fetch failed: {e}")

    # Inject sentiment as a static column (same value for all rows — today's news)
    # This is a simplification; it only influences the today_data prediction
    stock['News_Sentiment'] = 0.0  # neutral for all training rows
    today_data = stock.tail(1).copy()
    today_data['News_Sentiment'] = news_sentiment  # override with live score for prediction

    train_data = stock.dropna().copy()

    features = [
        'SMA_50', 'Today_Pct_Change', 'RSI', 'Vol_Change', 'Daily_Range',
        'Dist_From_200', 'MACD', 'Dist_BB_Upper', 'Day_Of_Week',
        'Lag_1', 'Lag_2', 'Gap_Pct',
        'VIX_Close', 'VIX_Change',
        'News_Sentiment'
    ]

    # --- ENGINE SWAP ---
    ml_engine, model_name = build_model(model)
    ml_engine.fit(train_data[features], train_data['Target_Future_Pct'])

    if model == "rf":
        r_squared = ml_engine.oob_score_
    else:
        r_squared = ml_engine.score(train_data[features], train_data['Target_Future_Pct'])

    prediction = ml_engine.predict(today_data[features])
    pred_val = prediction.item()
    price_val = today_data['Close'].item()

    fit_score = max(0, round(float(r_squared) * 100, 1))

    # --- WALK-FORWARD DIRECTIONAL ACCURACY (the real confidence) ---
    directional_accuracy, backtest_samples = walk_forward_directional_accuracy(
        train_data, features, 'Target_Future_Pct', model
    )

    # --- INFO PANEL DATA ---
    training_start = train_data.index[0].strftime('%Y-%m-%d')
    training_end = train_data.index[-1].strftime('%Y-%m-%d')
    num_training_rows = len(train_data)
    oob_note = round(float(ml_engine.oob_score_) * 100, 1) if model == "rf" else None

    return {
        "symbol": symbol.upper(),
        "predicted_change_pct": round(float(pred_val), 3),
        "current_price": round(float(price_val), 2),
        "confidence": directional_accuracy if directional_accuracy is not None else fit_score,
        "directional_accuracy": directional_accuracy,
        "backtest_samples": backtest_samples,
        "fit_score": fit_score,
        "model_used": model_name,
        "horizon_days": horizon,
        # Info panel fields
        "training_rows": num_training_rows,
        "training_start": training_start,
        "training_end": training_end,
        "oob_score": oob_note,
        "features_used": len(features),
        "news_sentiment": news_sentiment,
        "headlines": headlines
    }