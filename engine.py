import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------
# 1. GET THE DATA (Python's version of tq_get)
# ---------------------------------------------------------
print("Fetching SPY data...")
# yfinance downloads the data directly into a Pandas Dataframe
spy = yf.download('SPY', start='2020-01-01', end='2024-01-01')

# ---------------------------------------------------------
# 2. FEATURE ENGINEERING (Python's version of mutate)
# ---------------------------------------------------------
print("Calculating features...")
# In pandas, we just declare new columns like this: df['New_Column'] = ...

# Feature 1: The 50-Day Simple Moving Average
spy['SMA_50'] = spy['Close'].rolling(window=50).mean()

# Feature 2: Today's Percent Change
# pct_change() does the math: (Today - Yesterday) / Yesterday
spy['Today_Pct_Change'] = spy['Close'].pct_change() * 100

# THE TARGET (Y): Tomorrow's Percent Change
# shift(-1) is the exact equivalent of R's lead() function. It pulls tomorrow's data up to today.
spy['Target_NextDay_Pct'] = spy['Today_Pct_Change'].shift(-1)

# Drop any rows with missing data (like the first 50 days for the SMA, and the last row)
spy = spy.dropna()

# ---------------------------------------------------------
# 3. DATA PARTITIONING (80/20 Split)
# ---------------------------------------------------------
# iloc is Python's way of slicing arrays, just like [rows, columns] in R
split_idx = int(len(spy) * 0.8)

train_data = spy.iloc[:split_idx]
test_data = spy.iloc[split_idx:]

# Define our X (Predictors) and Y (Target)
features = ['SMA_50', 'Today_Pct_Change']
X_train = train_data[features]
Y_train = train_data['Target_NextDay_Pct']

# ---------------------------------------------------------
# 4. FIT THE MODEL (Python's version of lm)
# ---------------------------------------------------------
print("Training the model...")
model = LinearRegression()
model.fit(X_train, Y_train)

# ---------------------------------------------------------
# 5. MAKE A PREDICTION!
# ---------------------------------------------------------
# Let's grab the very last day in our dataset to see what it predicts for "tomorrow"
last_day_features = test_data[features].tail(1)

# The predict function works just like it did in R
prediction = model.predict(last_day_features)

print("\n--- RESULTS ---")
print(f"Predicted % change for the next day: {prediction[0]:.3f}%")

# 2nd
from sklearn.metrics import mean_absolute_error

# ---------------------------------------------------------
# 6. ASSESSING ACCURACY (The Reality Check)
# ---------------------------------------------------------
print("\nCalculating accuracy on unseen data...")

# Ask the model to predict the percent change for every day in the test set
test_predictions = model.predict(test_data[features])

# Compare the predictions to what actually happened
mae = mean_absolute_error(test_data['Target_NextDay_Pct'], test_predictions)

print(f"Mean Absolute Error (Test Set): {mae:.3f}%")