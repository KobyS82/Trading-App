from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

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

    # 1. Download target symbol and VIX
    stock = yf.download(symbol, period='max', progress=False, auto_adjust=True).tail(2500)
    vix = yf.download('^VIX', period='max', progress=False, auto_adjust=True).tail(2500)

    # 2. Clean up MultiIndex columns
    if isinstance(stock.columns, pd.MultiIndex):
        stock.columns = stock.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    # 3. Inject VIX
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

    today_data = stock.tail(1).copy()
    train_data = stock.dropna().copy()

    features = [
        'SMA_50', 'Today_Pct_Change', 'RSI', 'Vol_Change', 'Daily_Range',
        'Dist_From_200', 'MACD', 'Dist_BB_Upper', 'Day_Of_Week',
        'Lag_1', 'Lag_2', 'Gap_Pct',
        'VIX_Close', 'VIX_Change'
    ]

    # --- ENGINE SWAP ---
    if model == "rf":
        ml_engine = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            oob_score=True,
            max_depth=6,
            min_samples_leaf=5,
            max_features="sqrt"
        )
        model_name = "Random Forest"
    else:
        ml_engine = LinearRegression()
        model_name = "Linear Regression"

    ml_engine.fit(train_data[features], train_data['Target_Future_Pct'])

    if model == "rf":
        r_squared = ml_engine.oob_score_
    else:
        r_squared = ml_engine.score(train_data[features], train_data['Target_Future_Pct'])

    prediction = ml_engine.predict(today_data[features])

    pred_val = prediction.item()
    price_val = today_data['Close'].item()

    final_confidence = max(0, round(float(r_squared) * 100, 1))

    # --- INFO PANEL DATA ---
    training_start = train_data.index[0].strftime('%Y-%m-%d')
    training_end = train_data.index[-1].strftime('%Y-%m-%d')
    num_training_rows = len(train_data)

    oob_note = round(float(ml_engine.oob_score_) * 100, 1) if model == "rf" else None

    return {
        "symbol": symbol.upper(),
        "predicted_change_pct": round(float(pred_val), 3),
        "current_price": round(float(price_val), 2),
        "confidence": final_confidence,
        "model_used": model_name,
        "horizon_days": horizon,
        # Info panel fields
        "training_rows": num_training_rows,
        "training_start": training_start,
        "training_end": training_end,
        "oob_score": oob_note,
        "features_used": len(features)
    }