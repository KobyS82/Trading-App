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

# Notice we added model_type here! It defaults to "linear"
@app.get("/predict")
def get_prediction(model_type: str = "linear"):
    print(f"API called: Running {model_type} prediction engine...")
    
    spy = yf.download('SPY', period='max', progress=False, auto_adjust=True)
    spy = spy.tail(2500) 
    
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    
    # Feature Engineering
    spy['SMA_50'] = spy['Close'].rolling(window=50).mean()
    spy['Today_Pct_Change'] = spy['Close'].pct_change() * 100
    spy['Target_NextDay_Pct'] = spy['Today_Pct_Change'].shift(-1)
    spy['Vol_Change'] = spy['Volume'].pct_change() * 100
    spy['Daily_Range'] = (spy['High'] - spy['Low']) / spy['Close'] * 100
    spy['SMA_200'] = spy['Close'].rolling(window=200).mean()
    spy['Dist_From_200'] = (spy['Close'] - spy['SMA_200']) / spy['SMA_200'] * 100

    delta = spy['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    spy['RSI'] = 100 - (100 / (1 + rs))
    
    today_data = spy.tail(1).copy() 
    train_data = spy.dropna().copy()
    
    features = ['SMA_50', 'Today_Pct_Change', 'RSI', 'Vol_Change', 'Daily_Range', 'Dist_From_200']
    
    # --- SWAP ENGINE BASED ON URL PARAMETER ---
    if model_type == "rf":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model_name = "Random Forest"
    else:
        model = LinearRegression()
        model_name = "Linear Regression"
        
    model.fit(train_data[features], train_data['Target_NextDay_Pct'])
    r_squared = model.score(train_data[features], train_data['Target_NextDay_Pct'])
    prediction = model.predict(today_data[features])
    
    pred_val = prediction.item() 
    price_val = today_data['Close'].item()
    
    return {
        "symbol": "SPY",
        "predicted_change_pct": round(float(pred_val), 3),
        "current_price": round(float(price_val), 2),
        "confidence": round(float(r_squared) * 100, 1),
        "model_used": model_name
    }