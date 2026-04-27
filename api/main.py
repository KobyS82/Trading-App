from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression

# Initialize the web app
app = FastAPI()

# This middleware allows your local JS frontend to talk to this Python backend without security blocks (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create an "endpoint" that your web app can fetch data from
@app.get("/predict")
def get_prediction():
    print("API called: Running prediction engine...")
    
    # 1. Pull data
    spy = yf.download('SPY', period='1y', progress=False, auto_adjust=True)
    
    # This flattens the columns so 'Close' is just a simple string again
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    
    # 2. Engineer features
    spy['SMA_50'] = spy['Close'].rolling(window=50).mean()
    spy['Today_Pct_Change'] = spy['Close'].pct_change() * 100
    spy['Target_NextDay_Pct'] = spy['Today_Pct_Change'].shift(-1)
    
    # --- Updated Feature Engineering in main.py ---
    # Calculate Price Gains and Losses for RSI
    delta = spy['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    spy['RSI'] = 100 - (100 / (1 + rs))
    
    today_data = spy.tail(1).copy() 
    train_data = spy.dropna().copy()
    
    # 3. Train the model
    features = ['SMA_50', 'Today_Pct_Change', 'RSI']
    model = LinearRegression()
    model.fit(train_data[features], train_data['Target_NextDay_Pct'])
    
    # This returns a value between 0 and 1
    r_squared = model.score(train_data[features], train_data['Target_NextDay_Pct'])

    # 4. Predict
    prediction = model.predict(today_data[features])
    
    # --- USE .item() HERE ---
    # .item() is a special function that says: 
    # "I don't care if you're a Series, an Array, or a List—if you only have ONE value, give it to me as a native number."
    pred_val = prediction.item() 
    price_val = today_data['Close'].item()
    
    return {
        "symbol": "SPY",
        "predicted_change_pct": round(float(pred_val), 3),
        "current_price": round(float(price_val), 2),
        "confidence": round(float(r_squared) * 100, 1)
    }