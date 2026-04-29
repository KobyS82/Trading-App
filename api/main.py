from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd
import httpx
import math
import os
import time
import threading
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from textblob import TextBlob

_scan_lock = threading.Lock()

load_dotenv()  # loads .env on local dev; no-op on Render (uses dashboard env vars)


@asynccontextmanager
async def lifespan(_app):
    _scheduler.add_job(auto_scan,               CronTrigger(day_of_week="mon-fri", hour=9,  minute=35))
    _scheduler.add_job(auto_scan,               CronTrigger(day_of_week="mon-fri", hour=12, minute=30))
    _scheduler.add_job(_check_paper_trades_job, CronTrigger(day_of_week="mon-fri", hour=15, minute=55))
    _scheduler.start()
    print("[scheduler] Started — scans at 09:35, 12:30, and 15:55 ET, Mon–Fri")
    yield
    _scheduler.shutdown()


app = FastAPI(lifespan=lifespan)

# ── SUPABASE CONFIG ──────────────────────────────────────────────────────────
# Set SUPABASE_URL and SUPABASE_ANON_KEY in .env locally, Render dashboard in prod.
# Use the Legacy Anon JWT key (eyJ...) — NOT the secret key, which bypasses all RLS.
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY", "")


def _sb_headers() -> dict:
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }


def _log_to_supabase(data: dict) -> None:
    """Background task: insert a prediction record. Silently skips if Supabase is not configured."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return
    try:
        payload = {
            "symbol":               data["symbol"],
            "model":                data["model_used"],
            "horizon":              data["horizon_days"],
            "signal":               data["signal"],
            "conviction":           data["conviction"],
            "predicted_pct":        data["predicted_change_pct"],
            "entry_price":          data["current_price"],
            "directional_accuracy": data["directional_accuracy"],
        }
        with httpx.Client(timeout=6.0) as client:
            client.post(
                f"{SUPABASE_URL}/rest/v1/predictions",
                headers={**_sb_headers(), "Prefer": "return=minimal"},
                json=payload,
            )
    except Exception as exc:
        print(f"Supabase log failed: {exc}")

# In-memory cache: (symbol, model, horizon) -> (timestamp, response)
# Prevents the watchlist scanner from re-running expensive ML for every tab load.
_predict_cache: dict = {}
_CACHE_TTL_SEC = 30 * 60

# ── PAPER TRADING BOT ────────────────────────────────────────────────────────

SCAN_WATCHLIST = [
    # Mega-cap tech
    "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","AMD","INTC","AVGO",
    "QCOM","MU","ORCL","CRM","ADBE","NOW","SNOW","PLTR","COIN","MSTR",

    # Financials
    "JPM","GS","MS","BAC","V","MA","PYPL","BRK-B",

    # Consumer / Retail
    "WMT","HD","COST","TGT","AMZN","NKE","MCD","SBUX",

    # Healthcare
    "JNJ","UNH","PFE","ABBV","LLY","MRK",

    # Energy / Industrials
    "XOM","CVX","BA","CAT","GE",

    # High volatility / retail favorites
    "NFLX","DIS","UBER","LYFT","RIVN","SOFI",

    # ETFs / Indexes
    "SPY","QQQ","IWM","DIA","GLD","SLV","XLF","XLE","XLK","ARKK",
]

# Per-horizon entry thresholds and trade parameters
SCAN_CRITERIA = [
    {"days":  1, "min_da": 60.0, "convictions": {"Strong"},             "stop":  -3.0, "target":  4.0},
    {"days":  3, "min_da": 55.0, "convictions": {"Strong", "Moderate"}, "stop":  -4.0, "target":  6.0},
    {"days":  5, "min_da": 55.0, "convictions": {"Strong", "Moderate"}, "stop":  -5.0, "target":  8.0},
    {"days": 10, "min_da": 52.0, "convictions": {"Strong", "Moderate"}, "stop":  -6.0, "target": 10.0},
    {"days": 21, "min_da": 52.0, "convictions": {"Strong", "Moderate"}, "stop":  -8.0, "target": 15.0},
]

PAPER_NOTIONAL = 1000.0  # fixed USD notional per paper trade — keeps all trades comparable

_scheduler = BackgroundScheduler(timezone=pytz.timezone("America/New_York"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fed decision announcement dates (second day of each 2-day meeting)
FOMC_DATES = [
    date(2025, 3, 19), date(2025, 5, 7),  date(2025, 6, 18),
    date(2025, 7, 30), date(2025, 9, 17), date(2025, 10, 29), date(2025, 12, 10),
    date(2026, 1, 28), date(2026, 3, 18), date(2026, 5, 6),  date(2026, 6, 17),
    date(2026, 7, 29), date(2026, 9, 16), date(2026, 10, 28), date(2026, 12, 9),
]


def is_near_fomc(days_window=3):
    today = date.today()
    return any(abs((f - today).days) <= days_window for f in FOMC_DATES)


def build_model(name: str):
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


_ROLLING_TRAIN_WINDOW = 1000  # ~4 years; keeps the model in the current regime


def walk_forward_directional_accuracy(train_data, features, target_col, model_name,
                                      n_windows=5, test_size=80):
    correct = 0
    total = 0
    history = []
    n = len(train_data)
    for i in range(n_windows):
        end_idx = n - i * test_size
        start_idx = end_idx - test_size
        if start_idx < 250:
            break
        train = train_data.iloc[max(0, start_idx - _ROLLING_TRAIN_WINDOW):start_idx]
        test  = train_data.iloc[start_idx:end_idx]
        m, _ = build_model(model_name)
        m.fit(train[features], train[target_col])
        preds   = m.predict(test[features])
        actuals = test[target_col].values
        correct += int(((preds >= 0) == (actuals >= 0)).sum())
        total   += len(preds)
        for j in range(len(preds)):
            if pd.isna(actuals[j]):
                continue
            history.append({
                "date":          test.index[j].strftime('%Y-%m-%d'),
                "predicted_pct": round(float(preds[j]),   2),
                "actual_pct":    round(float(actuals[j]), 2),
                "correct":       bool((preds[j] >= 0) == (actuals[j] >= 0)),
            })
    if total == 0:
        return None, 0, []
    history.sort(key=lambda x: x["date"], reverse=True)
    return round(correct / total * 100, 1), total, history


def get_consensus_directions(train_data, features, target_col, today_data,
                              selected_model, selected_pred):
    """Fit all 3 models (light settings for the non-selected ones) and return direction dict."""
    results = {}
    for name, label in [("linear", "Linear"), ("rf", "Random Forest"), ("lgb", "LightGBM")]:
        if name == selected_model:
            p = selected_pred
        elif name == "rf":
            m = RandomForestRegressor(n_estimators=60, random_state=42, max_depth=5)
            m.fit(train_data[features], train_data[target_col])
            p = m.predict(today_data[features]).item()
        elif name == "lgb":
            m = LGBMRegressor(n_estimators=100, learning_rate=0.05,
                               random_state=42, n_jobs=1, verbose=-1)
            m.fit(train_data[features], train_data[target_col])
            p = m.predict(today_data[features]).item()
        else:
            m = LinearRegression()
            m.fit(train_data[features], train_data[target_col])
            p = m.predict(today_data[features]).item()
        results[label] = "up" if p >= 0 else "down"
    return results


def _open_trade_exists(symbol: str, horizon_days: int) -> bool:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return False
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(
                f"{SUPABASE_URL}/rest/v1/paper_trades",
                headers=_sb_headers(),
                params={
                    "select":       "id",
                    "symbol":       f"eq.{symbol}",
                    "horizon_days": f"eq.{horizon_days}",
                    "status":       "eq.open",
                    "limit":        "1",
                },
            )
            return len(resp.json()) > 0
    except Exception:
        return False


def _insert_paper_trade(symbol, signal, horizon_days, entry_price, predicted_pct,
                        conviction, da_score, stop_loss_pct, take_profit_pct) -> None:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return
    try:
        with httpx.Client(timeout=6.0) as client:
            client.post(
                f"{SUPABASE_URL}/rest/v1/paper_trades",
                headers={**_sb_headers(), "Prefer": "return=minimal"},
                json={
                    "symbol":          symbol,
                    "signal":          signal,
                    "horizon_days":    horizon_days,
                    "entry_price":     entry_price,
                    "notional":        PAPER_NOTIONAL,
                    "predicted_pct":   predicted_pct,
                    "conviction":      conviction,
                    "da_score":        da_score,
                    "stop_loss_pct":   stop_loss_pct,
                    "take_profit_pct": take_profit_pct,
                    "status":          "open",
                },
            )
        print(f"[bot] Paper trade: {symbol} {signal} {horizon_days}d @ {entry_price}")
    except Exception as exc:
        print(f"[bot] Insert failed {symbol}: {exc}")


def _scan_ticker(symbol: str, vix_data: pd.DataFrame) -> list:
    """Download data once for a ticker, evaluate 1/3/5-day horizons, return qualifying trades."""
    import gc
    trades = []
    try:
        stock = yf.download(symbol, period="max", progress=False, auto_adjust=True).tail(2500)

        if isinstance(stock.columns, pd.MultiIndex):
            stock.columns = stock.columns.get_level_values(0)

        if len(stock) < 300:
            return trades

        stock["VIX_Close"]  = vix_data["Close"]
        stock["VIX_Change"] = vix_data["Close"].pct_change() * 100

        stock["SMA_50"]           = stock["Close"].rolling(50).mean()
        stock["Today_Pct_Change"] = stock["Close"].pct_change() * 100
        stock["Vol_Change"]       = stock["Volume"].pct_change() * 100
        stock["Daily_Range"]      = (stock["High"] - stock["Low"]) / stock["Close"] * 100
        stock["SMA_200"]          = stock["Close"].rolling(200).mean()
        stock["Dist_From_200"]    = (stock["Close"] - stock["SMA_200"]) / stock["SMA_200"] * 100

        delta = stock["Close"].diff()
        gain  = delta.where(delta > 0, 0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
        stock["RSI"] = 100 - (100 / (1 + gain / loss))

        ema_12 = stock["Close"].ewm(span=12, adjust=False).mean()
        ema_26 = stock["Close"].ewm(span=26, adjust=False).mean()
        stock["MACD"] = ema_12 - ema_26

        bb_mid   = stock["Close"].rolling(20).mean()
        bb_std   = stock["Close"].rolling(20).std()
        bb_upper = bb_mid + bb_std * 2
        stock["Dist_BB_Upper"] = (stock["Close"] - bb_upper) / bb_upper * 100

        stock["Day_Of_Week"] = stock.index.dayofweek
        stock["Lag_1"]       = stock["Today_Pct_Change"].shift(1)
        stock["Lag_2"]       = stock["Today_Pct_Change"].shift(2)
        prev_close           = stock["Close"].shift(1)
        stock["Gap_Pct"]     = (stock["Open"] - prev_close) / prev_close * 100

        stock["Ret_5d"]  = stock["Close"].pct_change(5)  * 100
        stock["Ret_10d"] = stock["Close"].pct_change(10) * 100
        stock["Ret_20d"] = stock["Close"].pct_change(20) * 100
        stock["Ret_63d"] = stock["Close"].pct_change(63) * 100

        stock["Vol_20d"]   = stock["Today_Pct_Change"].rolling(20).std()
        stock["Above_200"] = (stock["Close"] > stock["SMA_200"]).astype(int)

        stock["Earnings_Flag"] = 0
        near_earnings = False
        try:
            edates = yf.Ticker(symbol).earnings_dates
            if edates is not None and not edates.empty:
                idx_naive   = stock.index.tz_localize(None) if stock.index.tz else stock.index
                today_naive = pd.Timestamp.today().normalize()
                for edate in edates.index:
                    e = pd.Timestamp(edate).tz_localize(None) if pd.Timestamp(edate).tz else pd.Timestamp(edate)
                    mask = (idx_naive >= e - pd.Timedelta(days=5)) & (idx_naive <= e + pd.Timedelta(days=2))
                    stock.loc[mask, "Earnings_Flag"] = 1
                    if abs((e.normalize() - today_naive).days) <= 5:
                        near_earnings = True
        except Exception:
            pass

        stock["News_Sentiment"] = 0.0
        fomc_flag  = is_near_fomc(days_window=3)
        price_val  = float(stock["Close"].iloc[-1])
        today_data = stock.tail(1).copy()

        features = [
            "SMA_50", "Today_Pct_Change", "RSI", "Vol_Change", "Daily_Range",
            "Dist_From_200", "MACD", "Dist_BB_Upper", "Day_Of_Week",
            "Lag_1", "Lag_2", "Gap_Pct",
            "VIX_Close", "VIX_Change",
            "Ret_5d", "Ret_10d", "Ret_20d", "Ret_63d",
            "Vol_20d", "Above_200",
            "Earnings_Flag",
            "News_Sentiment",
        ]

        for h_cfg in SCAN_CRITERIA:
            horizon_days = h_cfg["days"]
            stock_h = stock.copy()
            stock_h["Target_Future_Pct"] = (
                stock_h["Close"].pct_change(horizon_days).shift(-horizon_days) * 100
            )
            train_data = stock_h.dropna().copy()
            if len(train_data) < 300:
                continue

            da, _, _ = walk_forward_directional_accuracy(
                train_data, features, "Target_Future_Pct", "lgb"
            )
            if da is None or da < h_cfg["min_da"]:
                continue

            m_main, _ = build_model("lgb")
            m_main.fit(train_data[features], train_data["Target_Future_Pct"])
            pred_val = m_main.predict(today_data[features]).item()

            if abs(pred_val) < 0.05:
                continue

            primary_dir = "up" if pred_val >= 0 else "down"
            ml_agree = 1  # lgb already agrees with itself
            m_rf = RandomForestRegressor(n_estimators=60, random_state=42, max_depth=5)
            m_rf.fit(train_data[features], train_data["Target_Future_Pct"])
            if ("up" if m_rf.predict(today_data[features]).item() >= 0 else "down") == primary_dir:
                ml_agree += 1

            if near_earnings or fomc_flag:
                continue
            elif da >= 55 and ml_agree == 2:
                conviction = "Strong"
            elif da >= 52 and ml_agree >= 1:
                conviction = "Moderate"
            else:
                continue

            if conviction not in h_cfg["convictions"]:
                continue
            if _open_trade_exists(symbol, horizon_days):
                continue

            trades.append({
                "symbol":          symbol,
                "signal":          "BUY" if pred_val > 0 else "SELL",
                "horizon_days":    horizon_days,
                "entry_price":     price_val,
                "predicted_pct":   round(pred_val, 3),
                "conviction":      conviction,
                "da_score":        da,
                "stop_loss_pct":   h_cfg["stop"],
                "take_profit_pct": h_cfg["target"],
            })

    except Exception as exc:
        print(f"[bot] Scan error {symbol}: {exc}")
    finally:
        try:
            del stock
        except Exception:
            pass
        gc.collect()

    return trades


def _check_paper_trades_job() -> dict:
    """Close open paper trades that have hit their stop loss, take profit, or elapsed horizon."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return {"updated": 0}

    updated = 0
    now = datetime.now(timezone.utc)

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(
                f"{SUPABASE_URL}/rest/v1/paper_trades",
                headers=_sb_headers(),
                params={
                    "select": "id,symbol,signal,horizon_days,entry_price,notional,stop_loss_pct,take_profit_pct,opened_at",
                    "status": "eq.open",
                    "limit":  "200",
                },
            )
            open_trades = resp.json()
    except Exception as exc:
        print(f"[bot] Check-trades fetch failed: {exc}")
        return {"updated": 0}

    for trade in open_trades:
        try:
            hist = yf.Ticker(trade["symbol"]).history(period="2d")
            if hist.empty:
                continue
            current_price = float(hist["Close"].iloc[-1])
            entry_price   = float(trade["entry_price"])
            raw_pct       = (current_price - entry_price) / entry_price * 100
            # outcome_pct is always from the trade's perspective: positive = winning
            outcome_pct   = raw_pct if trade["signal"] == "BUY" else -raw_pct
            outcome_pnl   = round(float(trade["notional"]) * outcome_pct / 100, 2)

            opened_at = datetime.fromisoformat(trade["opened_at"].replace("Z", "+00:00"))
            cal_days  = math.ceil(trade["horizon_days"] * 365 / 252)
            stopped   = outcome_pct <= trade["stop_loss_pct"]
            targeted  = outcome_pct >= trade["take_profit_pct"]
            expired   = (now - opened_at).days >= cal_days

            if not (stopped or targeted or expired):
                continue

            close_reason = "stop_loss" if stopped else ("take_profit" if targeted else "expired")

            with httpx.Client(timeout=5.0) as client:
                client.patch(
                    f"{SUPABASE_URL}/rest/v1/paper_trades",
                    headers={**_sb_headers(), "Prefer": "return=minimal"},
                    params={"id": f"eq.{trade['id']}"},
                    json={
                        "status":       close_reason,
                        "outcome_pct":  round(outcome_pct, 3),
                        "outcome_pnl":  outcome_pnl,
                        "closed_at":    now.isoformat(),
                        "close_reason": close_reason,
                    },
                )
            updated += 1
            print(f"[bot] Closed {trade['symbol']} {trade['signal']} → {close_reason} ({outcome_pct:+.1f}%)")
        except Exception as exc:
            print(f"[bot] Trade close error {trade.get('id')}: {exc}")

    return {"updated": updated}


def auto_scan() -> dict:
    if not _scan_lock.acquire(blocking=False):
        print("[bot] Scan already running, skipping")
        return {"entered": 0, "skipped": "scan already in progress"}
    try:
        print(f"[bot] Auto-scan start {datetime.now(timezone.utc).isoformat()}")
        _check_paper_trades_job()
        entered = 0
        # Download VIX once and reuse — saves 31 redundant downloads per scan
        vix_data = yf.download("^VIX", period="5y", progress=False, auto_adjust=True).tail(2500)
        if isinstance(vix_data.columns, pd.MultiIndex):
            vix_data.columns = vix_data.columns.get_level_values(0)
        for symbol in SCAN_WATCHLIST:
            try:
                for t in _scan_ticker(symbol, vix_data):
                    _insert_paper_trade(**t)
                    entered += 1
            except Exception as exc:
                print(f"[bot] Error {symbol}: {exc}")
            time.sleep(0.3)
        print(f"[bot] Auto-scan done — {entered} new paper trades")
        return {"entered": entered}
    finally:
        _scan_lock.release()


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/predict")
def get_prediction(
    background_tasks: BackgroundTasks,
    model: str = "lgb",
    horizon: int = 1,
    symbol: str = "SPY",
    source: str = "web",   # "screener" suppresses logging
):
    print(f"API called: {model} | {symbol} | {horizon}d | src={source}")

    # --- CACHE CHECK ---
    cache_key = (symbol.upper(), model, horizon)
    cached = _predict_cache.get(cache_key)
    if cached and (time.time() - cached[0]) < _CACHE_TTL_SEC:
        print(f"Cache hit: {cache_key}")
        return cached[1]

    # --- DATA DOWNLOAD ---
    stock = yf.download(symbol, period='max', progress=False, auto_adjust=True).tail(2500)
    vix   = yf.download('^VIX', period='max', progress=False, auto_adjust=True).tail(2500)

    if isinstance(stock.columns, pd.MultiIndex):
        stock.columns = stock.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    stock['VIX_Close']  = vix['Close']
    stock['VIX_Change'] = vix['Close'].pct_change() * 100

    # --- FEATURE ENGINEERING ---
    stock['SMA_50']          = stock['Close'].rolling(50).mean()
    stock['Today_Pct_Change']= stock['Close'].pct_change() * 100
    stock['Target_Future_Pct']= stock['Close'].pct_change(periods=horizon).shift(-horizon) * 100
    stock['Vol_Change']      = stock['Volume'].pct_change() * 100
    stock['Daily_Range']     = (stock['High'] - stock['Low']) / stock['Close'] * 100
    stock['SMA_200']         = stock['Close'].rolling(200).mean()
    stock['Dist_From_200']   = (stock['Close'] - stock['SMA_200']) / stock['SMA_200'] * 100

    delta = stock['Close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    stock['RSI'] = 100 - (100 / (1 + gain / loss))

    ema_12 = stock['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = stock['Close'].ewm(span=26, adjust=False).mean()
    stock['MACD'] = ema_12 - ema_26

    bb_mid = stock['Close'].rolling(20).mean()
    bb_std = stock['Close'].rolling(20).std()
    bb_upper = bb_mid + bb_std * 2
    stock['Dist_BB_Upper'] = (stock['Close'] - bb_upper) / bb_upper * 100

    stock['Day_Of_Week'] = stock.index.dayofweek
    stock['Lag_1']       = stock['Today_Pct_Change'].shift(1)
    stock['Lag_2']       = stock['Today_Pct_Change'].shift(2)
    prev_close           = stock['Close'].shift(1)
    stock['Gap_Pct']     = (stock['Open'] - prev_close) / prev_close * 100

    # Multi-timeframe momentum
    stock['Ret_5d']  = stock['Close'].pct_change(5)  * 100
    stock['Ret_10d'] = stock['Close'].pct_change(10) * 100
    stock['Ret_20d'] = stock['Close'].pct_change(20) * 100
    stock['Ret_63d'] = stock['Close'].pct_change(63) * 100  # quarterly — regime context

    # Regime indicators (help at longer horizons like 21+ DTE)
    stock['Vol_20d']   = stock['Today_Pct_Change'].rolling(20).std()  # realized vol regime
    stock['Above_200'] = (stock['Close'] > stock['SMA_200']).astype(int)  # bull/bear regime

    # --- EARNINGS PROXIMITY FLAG (historical) + NEXT EARNINGS DATE ---
    stock['Earnings_Flag'] = 0
    next_earnings = None
    ticker_obj = yf.Ticker(symbol)
    try:
        edates = ticker_obj.earnings_dates
        if edates is not None and not edates.empty:
            # Normalize both sides to tz-naive date for comparison
            idx = stock.index
            idx_naive = idx.tz_localize(None) if idx.tz is not None else idx
            today_naive = pd.Timestamp.today().normalize()
            future_edates = []
            for edate in edates.index:
                e = pd.Timestamp(edate).tz_localize(None) if pd.Timestamp(edate).tz is not None else pd.Timestamp(edate)
                mask = (idx_naive >= e - pd.Timedelta(days=5)) & (idx_naive <= e + pd.Timedelta(days=2))
                stock.loc[mask, 'Earnings_Flag'] = 1
                if e.normalize() > today_naive:
                    future_edates.append(e)
            if future_edates:
                next_earnings = min(future_edates).strftime('%Y-%m-%d')
    except Exception as ex:
        print(f"Earnings flag failed: {ex}")

    # --- NEWS + SENTIMENT ---
    headlines      = []
    news_sentiment = 0.0
    try:
        for item in ticker_obj.news[:5]:
            title = ""
            if isinstance(item.get('content'), dict):
                title = item['content'].get('title', '')
            elif isinstance(item.get('title'), str):
                title = item['title']
            if title:
                sentiment = TextBlob(title).sentiment.polarity
                link = ""
                if isinstance(item.get('content'), dict):
                    canonical = item['content'].get('canonicalUrl', {})
                    link = canonical.get('url', '') if isinstance(canonical, dict) else ''
                elif isinstance(item.get('link'), str):
                    link = item['link']
                headlines.append({"title": title, "sentiment": round(sentiment, 3), "url": link})
        if headlines:
            news_sentiment = round(sum(h['sentiment'] for h in headlines) / len(headlines), 4)
    except Exception as ex:
        print(f"News fetch failed: {ex}")

    # --- PUT/CALL RATIO + IMPLIED VOLATILITY (nearest expiry, display only) ---
    put_call_ratio = None
    implied_vol    = None
    try:
        opts = ticker_obj.options
        if opts:
            chain    = ticker_obj.option_chain(opts[0])
            put_vol  = chain.puts['volume'].fillna(0).sum()
            call_vol = chain.calls['volume'].fillna(0).sum()
            if call_vol > 0:
                put_call_ratio = round(float(put_vol / call_vol), 2)
            # ATM implied volatility — average of nearest call + put strike to current price
            cur_price = float(stock['Close'].iloc[-1])
            ivs = []
            for chain_side in [chain.calls, chain.puts]:
                if not chain_side.empty and 'impliedVolatility' in chain_side.columns:
                    pos = int((chain_side['strike'] - cur_price).abs().argsort().iloc[0])
                    iv  = chain_side.iloc[pos]['impliedVolatility']
                    if pd.notna(iv) and iv > 0:
                        ivs.append(float(iv))
            if ivs:
                implied_vol = round(sum(ivs) / len(ivs) * 100, 1)
    except Exception as ex:
        print(f"P/C ratio failed: {ex}")

    # --- FOMC FLAG ---
    fomc_flag = is_near_fomc(days_window=3)

    # --- INJECT TODAY OVERRIDES ---
    stock['News_Sentiment'] = 0.0
    today_data = stock.tail(1).copy()
    today_data['News_Sentiment'] = news_sentiment
    # Earnings flag already set correctly for today via the historical loop above

    train_data = stock.dropna().copy()

    features = [
        'SMA_50', 'Today_Pct_Change', 'RSI', 'Vol_Change', 'Daily_Range',
        'Dist_From_200', 'MACD', 'Dist_BB_Upper', 'Day_Of_Week',
        'Lag_1', 'Lag_2', 'Gap_Pct',
        'VIX_Close', 'VIX_Change',
        'Ret_5d', 'Ret_10d', 'Ret_20d', 'Ret_63d',
        'Vol_20d', 'Above_200',
        'Earnings_Flag',
        'News_Sentiment',
    ]

    # --- MAIN MODEL FIT ---
    ml_engine, model_name = build_model(model)
    ml_engine.fit(train_data[features], train_data['Target_Future_Pct'])

    r_squared = (ml_engine.oob_score_
                 if model == "rf"
                 else ml_engine.score(train_data[features], train_data['Target_Future_Pct']))

    prediction = ml_engine.predict(today_data[features])
    pred_val   = prediction.item()
    price_val  = today_data['Close'].item()
    fit_score  = max(0, round(float(r_squared) * 100, 1))

    # --- WALK-FORWARD DIRECTIONAL ACCURACY ---
    directional_accuracy, backtest_samples, prediction_history = walk_forward_directional_accuracy(
        train_data, features, 'Target_Future_Pct', model
    )

    # --- MODEL CONSENSUS ---
    consensus_results = get_consensus_directions(
        train_data, features, 'Target_Future_Pct', today_data, model, pred_val
    )
    primary_dir    = "up" if pred_val >= 0 else "down"
    models_agreeing = sum(1 for d in consensus_results.values() if d == primary_dir)
    # Conviction uses only the two ML models (Linear is too simple for noisy price data)
    ml_agreeing = sum(
        1 for label, d in consensus_results.items()
        if label in ("Random Forest", "LightGBM") and d == primary_dir
    )

    # --- SIGNAL + CONVICTION ---
    today_earnings = int(today_data['Earnings_Flag'].item())

    if abs(pred_val) < 0.05:
        signal = "HOLD"
    elif pred_val > 0:
        signal = "BUY"
    else:
        signal = "SELL"

    signal_note = None
    if today_earnings or fomc_flag:
        conviction  = "Weak"
        signal_note = "Near earnings or Fed meeting: elevated uncertainty"
    elif directional_accuracy and directional_accuracy >= 55 and ml_agreeing == 2:
        conviction = "Strong"
    elif directional_accuracy and directional_accuracy >= 52 and ml_agreeing >= 1:
        conviction = "Moderate"
    else:
        conviction  = "Weak"
        signal_note = "Low model agreement or sub-52% historical accuracy on this ticker/horizon"

    # --- INFO PANEL ---
    training_start    = train_data.index[0].strftime('%Y-%m-%d')
    training_end      = train_data.index[-1].strftime('%Y-%m-%d')
    num_training_rows = len(train_data)
    oob_note          = round(float(ml_engine.oob_score_) * 100, 1) if model == "rf" else None

    response = {
        "symbol":               symbol.upper(),
        "current_price":        round(float(price_val), 2),
        "predicted_change_pct": round(float(pred_val), 3),
        "horizon_days":         horizon,
        # Signal
        "signal":               signal,
        "conviction":           conviction,
        "signal_note":          signal_note,
        # Consensus
        "consensus_results":    consensus_results,
        "models_agreeing":      models_agreeing,
        # Context flags
        "earnings_flag":        bool(today_earnings),
        "fomc_flag":            fomc_flag,
        "put_call_ratio":       put_call_ratio,
        # Accuracy
        "directional_accuracy": directional_accuracy,
        "backtest_samples":     backtest_samples,
        "confidence":           directional_accuracy if directional_accuracy is not None else fit_score,
        "fit_score":            fit_score,
        "model_used":           model_name,
        # Info panel
        "training_rows":        num_training_rows,
        "training_start":       training_start,
        "training_end":         training_end,
        "oob_score":            oob_note,
        "features_used":        len(features),
        "news_sentiment":       news_sentiment,
        "headlines":            headlines,
        "implied_vol":          implied_vol,
        "next_earnings":        next_earnings,
        # Market context (for UI display)
        "vix_close": round(float(today_data['VIX_Close'].item()), 2) if pd.notna(today_data['VIX_Close'].item()) else None,
        "rsi":       round(float(today_data['RSI'].item()),       1) if pd.notna(today_data['RSI'].item())       else None,
        "macd":      round(float(today_data['MACD'].item()),      3) if pd.notna(today_data['MACD'].item())      else None,
        # Track record (walk-forward backtested predictions, most recent first)
        "prediction_history": prediction_history[:25],
    }

    _predict_cache[cache_key] = (time.time(), response)

    # Log to Supabase in the background (non-blocking, skips scanner calls)
    if source != "screener":
        background_tasks.add_task(_log_to_supabase, response)

    return response


# ── SUPABASE OUTCOME CHECKER ─────────────────────────────────────────────────

@app.get("/check-outcomes")
def check_outcomes():
    """
    Fetch prices for predictions whose horizon has elapsed and write actual_pct + was_correct.
    Safe to call repeatedly — skips rows that already have an outcome.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        return {"error": "Supabase not configured", "updated": 0}

    headers = _sb_headers()
    updated = 0
    errors  = []

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(
                f"{SUPABASE_URL}/rest/v1/predictions",
                headers=headers,
                params={
                    "select":      "id,symbol,horizon,predicted_pct,entry_price,logged_at",
                    "outcome_at":  "is.null",
                    "order":       "logged_at.asc",
                    "limit":       "100",
                },
            )
            pending = resp.json()
    except Exception as exc:
        return {"error": str(exc), "updated": 0}

    now = datetime.now(timezone.utc)
    for row in pending:
        try:
            logged_at = datetime.fromisoformat(row["logged_at"].replace("Z", "+00:00"))
            # horizon is in trading days; convert to calendar days conservatively
            cal_days_needed = math.ceil(row["horizon"] * 365 / 252)
            if (now - logged_at).days < cal_days_needed:
                continue  # not enough time has elapsed yet

            hist = yf.Ticker(row["symbol"]).history(period="2d")
            if hist.empty:
                continue
            current_price = float(hist["Close"].iloc[-1])
            actual_pct    = round((current_price - row["entry_price"]) / row["entry_price"] * 100, 3)
            was_correct   = (row["predicted_pct"] >= 0) == (actual_pct >= 0)

            with httpx.Client(timeout=5.0) as client:
                client.patch(
                    f"{SUPABASE_URL}/rest/v1/predictions",
                    headers={**headers, "Prefer": "return=minimal"},
                    params={"id": f"eq.{row['id']}"},
                    json={
                        "outcome_at":  now.isoformat(),
                        "actual_pct":  actual_pct,
                        "was_correct": was_correct,
                    },
                )
            updated += 1
        except Exception as exc:
            errors.append(f"{row.get('id')}: {exc}")

    return {"updated": updated, "pending_skipped": len(pending) - updated, "errors": errors[:5]}


@app.get("/logs")
def get_logs(limit: int = 100):
    """Return recent prediction log entries with outcomes where available."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return {"error": "Supabase not configured", "logs": []}
    try:
        with httpx.Client(timeout=8.0) as client:
            resp = client.get(
                f"{SUPABASE_URL}/rest/v1/predictions",
                headers=_sb_headers(),
                params={
                    "select": "*",
                    "order":  "logged_at.desc",
                    "limit":  str(min(limit, 500)),
                },
            )
        return {"logs": resp.json()}
    except Exception as exc:
        return {"error": str(exc), "logs": []}


@app.api_route("/scan-now", methods=["GET", "POST"])
def scan_now():
    """Trigger the paper trading scanner immediately in the background."""
    t = threading.Thread(target=auto_scan, daemon=True)
    t.start()
    return {"status": "scan started", "watchlist_size": len(SCAN_WATCHLIST)}


@app.api_route("/check-paper-trades", methods=["GET", "POST"])
def check_paper_trades_endpoint():
    """Resolve open paper trades that have hit stop/target/expiry."""
    return _check_paper_trades_job()


@app.get("/paper-trades")
def get_paper_trades(status: str = "all", limit: int = 100):
    """Return paper trades. status filter: 'open', 'stop_loss', 'take_profit', 'expired', or 'all'."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return {"error": "Supabase not configured", "trades": []}
    try:
        params = {
            "select": "*",
            "order":  "opened_at.desc",
            "limit":  str(min(limit, 500)),
        }
        if status != "all":
            params["status"] = f"eq.{status}"
        with httpx.Client(timeout=8.0) as client:
            resp = client.get(
                f"{SUPABASE_URL}/rest/v1/paper_trades",
                headers=_sb_headers(),
                params=params,
            )
        trades = resp.json()
        closed = [t for t in trades if t.get("status") != "open"]
        total_pnl   = round(sum(t.get("outcome_pnl") or 0 for t in closed), 2)
        wins        = sum(1 for t in closed if (t.get("outcome_pnl") or 0) > 0)
        win_rate    = round(wins / len(closed) * 100, 1) if closed else None
        return {
            "trades":    trades,
            "summary": {
                "total_trades":  len(trades),
                "open":          sum(1 for t in trades if t.get("status") == "open"),
                "closed":        len(closed),
                "total_pnl":     total_pnl,
                "win_rate_pct":  win_rate,
            },
        }
    except Exception as exc:
        return {"error": str(exc), "trades": []}
