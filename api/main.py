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
    spy = yf.download('SPY', period='max', progress=False, auto_adjust=True)
    # For performance, we can limit to the most recent 2500 rows (about 10 years of daily data)
    spy = spy.tail(2500)
    
    # This flattens the columns so 'Close' is just a simple string again
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    
    # 2. Engineer features
        # SMA_50: 50-day Simple Moving Average
    spy['SMA_50'] = spy['Close'].rolling(window=50).mean()
    spy['Today_Pct_Change'] = spy['Close'].pct_change() * 100
    spy['Target_NextDay_Pct'] = spy['Today_Pct_Change'].shift(-1)
    
        # Vol_Change: Volume Momentum (Today's Volume vs Yesterday's)
    spy['Vol_Change'] = spy['Volume'].pct_change() * 100
    
        # Daily_Range: (High - Low) as a percentage of Close
    spy['Daily_Range'] = (spy['High'] - spy['Low']) / spy['Close'] * 100
    
        # SMA_200 and Distance from it: A common long-term trend indicator
    spy['SMA_200'] = spy['Close'].rolling(window=200).mean()
    spy['Dist_From_200'] = (spy['Close'] - spy['SMA_200']) / spy['SMA_200'] * 100

    # Calculate Price Gains and Losses for RSI
    delta = spy['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    spy['RSI'] = 100 - (100 / (1 + rs))
    
    today_data = spy.tail(1).copy() 
    train_data = spy.dropna().copy()
    
    # 3. Train the model
    features = ['SMA_50', 'Today_Pct_Change', 'RSI', 'Vol_Change', 'Daily_Range', 'Dist_From_200']
    model = LinearRegression()
    model.fit(train_data[features], train_data['Target_NextDay_Pct'])
    
    # This returns a value between 0 and 1
    r_squared = model.score(train_data[features], train_data['Target_NextDay_Pct'])

    # 4. Predict
    prediction = model.predict(today_data[features])
    
    pred_val = prediction.item() 
    price_val = today_data['Close'].item()
    
    return {
        "symbol": "SPY",
        "predicted_change_pct": round(float(pred_val), 3),
        "current_price": round(float(price_val), 2),
        "confidence": round(float(r_squared) * 100, 1)
    }