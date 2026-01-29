# train_model.py - FIXED VERSION WITH ENHANCED DATA GENERATION
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, classification_report
import joblib
from datetime import datetime, timedelta

print("Starting model training...")

def generate_sample_stock_data(days=1260):
    """Generate realistic stock data for training (5 years = ~1260 trading days)"""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Generate more realistic price data with volatility clustering
    base_price = 400
    returns = np.random.randn(days) * 0.015  # Daily returns ~1.5% std
    
    # Add momentum effect (prices tend to continue in same direction)
    returns = pd.Series(returns).ewm(span=20).mean().values
    
    # Calculate cumulative prices from returns
    close_prices = base_price * np.exp(np.cumsum(returns))
    
    # Volume increases with volatility
    volume_base = 80000000
    volume = volume_base + np.abs(returns) * 500000000
    
    # Create OHLC data with realistic relationships
    data = pd.DataFrame({
        'Date': dates,
        'Open': close_prices * (1 + np.random.randn(days) * 0.002),
        'High': close_prices * (1 + np.abs(np.random.randn(days)) * 0.01),
        'Low': close_prices * (1 - np.abs(np.random.randn(days)) * 0.01),
        'Close': close_prices,
        'Volume': volume.astype(int)
    })
    
    # Ensure High >= Close and Low <= Close
    data['High'] = data[['High', 'Close', 'Open']].max(axis=1)
    data['Low'] = data[['Low', 'Close', 'Open']].min(axis=1)
    
    return data

def add_technical_indicators(df):
    """Add technical indicators matching main.py exactly"""
    df = df.copy()
    
    # 1. Moving Averages
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    
    # 2. Relative Strength Index (RSI)
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # 3. Trend & Volatility
    df["Trend"] = (df["Close"].shift(1) < df["Close"]).astype(int)
    df["Volatility"] = df["Close"].pct_change().rolling(10).std()
    
    # 4. MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    
    return df

try:
    # Check if CSV exists, otherwise generate data
    try:
        print("Checking for existing CSV data...")
        data = pd.read_csv("SPY_data.csv")
        data['Date'] = pd.to_datetime(data['Date'])
        print(f"✓ Loaded {len(data)} rows from SPY_data.csv")
    except FileNotFoundError:
        print("CSV not found. Generating sample training data...")
        data = generate_sample_stock_data(1260)  # 5 years of data
        data.to_csv("SPY_data.csv", index=False)
        print(f"✓ Generated {len(data)} rows of sample data and saved to SPY_data.csv")

    # --- 2. Create Features and Target ---
    print("Adding technical indicators...")
    df = add_technical_indicators(data)

    # Target: Will price be higher 5 days from now?
    df["Target"] = (df["Close"].shift(-5) > df["Close"]).astype(int)
    df = df.dropna()

    print(f"Data prepared: {len(df)} samples after cleaning")

    # --- 3. Define Features and Split Data ---
    # FIXED: Must match main.py exactly
    features = ["SMA_10", "SMA_50", "RSI", "Volatility", "MACD", "Trend", "Open", "High", "Low", "Close", "Volume"]
    
    X = df[features]
    y = df["Target"]

    # Check class distribution
    print(f"Target distribution: {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training model on {len(X_train)} samples...")

    # --- 4. Train the XGBoost Model ---
    model = XGBClassifier(
        n_estimators=100, 
        learning_rate=0.05, 
        max_depth=5, 
        random_state=1, 
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)

    # --- 5. Test and Save Model ---
    preds = model.predict(X_test)
    precision = precision_score(y_test, preds)
    
    print(f"\n{'='*50}")
    print(f"MODEL PERFORMANCE")
    print(f"{'='*50}")
    print(f"Precision on test data: {precision:.2%}")
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, preds, target_names=['Down', 'Up']))

    model_filename = "stock_xgb_model.joblib"
    joblib.dump(model, model_filename)

    print(f"\n✓ Model saved as '{model_filename}'")
    print(f"✓ Features used: {len(features)}")
    print(f"✓ Feature list: {features}")
    print("\n" + "="*50)
    print("✓✓✓ SUCCESS! You can now run main.py ✓✓✓")
    print("="*50)
    
except Exception as e:
    print(f"\n✗ Error occurred: {e}")
    import traceback
    traceback.print_exc()