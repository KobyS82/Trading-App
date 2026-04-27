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
def get_prediction(model: str = "linear"):
    print(f"API called: Running {model} prediction engine...")
    
    spy = yf.download('SPY', period='max', progress=False, auto_adjust=True)
    spy = spy.tail(2500) 
    
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    
    # --- FEATURE ENGINEERING ---
    spy['SMA_50'] = spy['Close'].rolling(window=50).mean()
    spy['Today_Pct_Change'] = spy['Close'].pct_change() * 100
    spy['Target_NextDay_Pct'] = spy['Today_Pct_Change'].shift(-1)
    spy['Vol_Change'] = spy['Volume'].pct_change() * 100
    spy['Daily_Range'] = (spy['High'] - spy['Low']) / spy['Close'] * 100
    spy['SMA_200'] = spy['Close'].rolling(window=200).mean()
    spy['Dist_From_200'] = (spy['Close'] - spy['SMA_200']) / spy['SMA_200'] * 100

    # RSI
    delta = spy['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    spy['RSI'] = 100 - (100 / (1 + rs))

    # MACD 
    ema_12 = spy['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = spy['Close'].ewm(span=26, adjust=False).mean()
    spy['MACD'] = ema_12 - ema_26
    
    # BBANDS 
    spy['BB_Middle'] = spy['Close'].rolling(window=20).mean()
    spy['BB_Std'] = spy['Close'].rolling(window=20).std()
    spy['BB_Upper'] = spy['BB_Middle'] + (spy['BB_Std'] * 2)
    spy['Dist_BB_Upper'] = (spy['Close'] - spy['BB_Upper']) / spy['BB_Upper'] * 100
    
    # Day_Of_Week 
    spy['Day_Of_Week'] = spy.index.dayofweek

    # Lagged Returns & Gap
    spy['Lag_1'] = spy['Today_Pct_Change'].shift(1)
    spy['Lag_2'] = spy['Today_Pct_Change'].shift(2)
    spy['Prev_Close'] = spy['Close'].shift(1)
    spy['Gap_Pct'] = (spy['Open'] - spy['Prev_Close']) / spy['Prev_Close'] * 100
    
    # Clean Data
    today_data = spy.tail(1).copy() 
    train_data = spy.dropna().copy()
    
    # Updated Features List
    features = [
        'SMA_50', 'Today_Pct_Change', 'RSI', 'Vol_Change', 'Daily_Range', 
        'Dist_From_200', 'MACD', 'Dist_BB_Upper', 'Day_Of_Week', 
        'Lag_1', 'Lag_2', 'Gap_Pct'
    ]
    
    # --- CLEAN ENGINE SWAP ---
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
    
    # 1. Train the assigned engine
    ml_engine.fit(train_data[features], train_data['Target_NextDay_Pct'])
    
    # 2. Calculate Reality-Checked Confidence
    if model == "rf":
        # Use the pop quiz score instead of the memorized score
        r_squared = ml_engine.oob_score_ 
    else:
        # Linear Regression doesn't have an OOB, but it's so rigid it rarely overfits anyway
        r_squared = ml_engine.score(train_data[features], train_data['Target_NextDay_Pct'])
        
    # 3. Predict tomorrow
    prediction = ml_engine.predict(today_data[features])
    
    pred_val = prediction.item() 
    price_val = today_data['Close'].item()
    
    # If OOB score is negative (meaning it's worse than just guessing the average), floor it to 0
    final_confidence = max(0, round(float(r_squared) * 100, 1))
    
    return {
        "symbol": "SPY",
        "predicted_change_pct": round(float(pred_val), 3),
        "current_price": round(float(price_val), 2),
        "confidence": final_confidence,
        "model_used": model_name
    }