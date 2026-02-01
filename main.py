import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import os
import joblib
from dotenv import load_dotenv

from google import genai

st.set_page_config(
    page_title="Hybrid AI Financial Agent",
    layout="wide"
)

load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    st.error("GOOGLE_API_KEY not found in environment variables.")
    st.stop()

MODEL_FILE = "stock_xgb_model.joblib"

if not os.path.exists(MODEL_FILE):
    st.error("Trained model file not found. Run train_model.py first.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

MODEL = load_model()
client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-pro"

st.session_state["model_name"] = GEMINI_MODEL
class StockPredictor:
    def __init__(self, ticker):
        self.ticker = ticker
        self.model = MODEL
        self.data = None

    def fetch_data(self, period="2y"):
        try:
            df = yf.Ticker(self.ticker).history(period=period)
            if df.empty:
                return False
            df = df.reset_index()
            df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
            self.data = df
            return True
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return False

    def add_indicators(self, df):
        df = df.copy()

        df["SMA_10"] = df["Close"].rolling(10).mean()
        df["SMA_50"] = df["Close"].rolling(50).mean()

        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        df["Trend"] = (df["Close"].shift(1) < df["Close"]).astype(int)
        df["Volatility"] = df["Close"].pct_change().rolling(10).std()

        exp1 = df["Close"].ewm(span=12, adjust=False).mean()
        exp2 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = exp1 - exp2

        return df.dropna()

    def predict(self):
        df = self.add_indicators(self.data)

        features = [
            "SMA_10", "SMA_50", "RSI", "Volatility",
            "MACD", "Trend", "Open", "High", "Low", "Close", "Volume"
        ]

        row = df.iloc[[-1]]
        prob = self.model.predict_proba(row[features])[0][1]

        signal = "HOLD"
        confidence = prob

        if prob > 0.55:
            signal = "BUY"
        elif prob < 0.45:
            signal = "SELL"
            confidence = 1 - prob

        return {
            "signal": signal,
            "confidence": round(confidence * 100, 1),
            "price": round(row["Close"].iloc[0], 2),
            "rsi": round(row["RSI"].iloc[0], 2),
            "macd": round(row["MACD"].iloc[0], 2),
            "sma50": round(row["SMA_50"].iloc[0], 2)
        }

def generate_analysis(ticker, price, signal, confidence, rsi, macd, sma50):
    prompt = f"""
You are a senior financial analyst.

Stock: {ticker}
Price: ${price}
Signal: {signal}
Confidence: {confidence}%
RSI: {rsi}
MACD: {macd}
SMA50: ${sma50}

Explain:
1. What the signal means
2. What the indicators suggest
3. Include a disclaimer (not financial advice)

Keep it short and simple.
"""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        return response.text

    except Exception:
    
        return f"""
###  Technical Summary

**Signal:** {signal}  
**Confidence:** {confidence}%

- **RSI:** {rsi} → {"Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"}
- **MACD:** {macd}
- **Price vs SMA50:** {"Above trend" if price > sma50 else "Below trend"}

 **Disclaimer:** This is NOT financial advice.
"""


# ----------------- STREAMLIT UI -----------------
st.title("Stock prediction agent")
#st.caption(f" Using Gemini model: {st.session_state['model_name']}")

col1, col2 = st.columns([2, 1])

with col1:
    ticker = st.text_input(
        "Enter Stock Ticker",
        placeholder="AAPL, TSLA, NVDA, RELIANCE.NS"
    )

with col2:
    st.write("")
    st.write("")
    analyze_btn = st.button("🔍 Analyze", use_container_width=True)

if analyze_btn:
    if not ticker:
        st.warning(" Please enter a stock ticker.")
        st.stop()

    predictor = StockPredictor(ticker.upper())

    with st.spinner("Analyzing..."):
        if not predictor.fetch_data():
            st.error(" Invalid ticker or no data available.")
            st.stop()

        result = predictor.predict()
        ai_analysis = generate_analysis(
            ticker, result["price"], result["signal"],
            result["confidence"], result["rsi"],
            result["macd"], result["sma50"]
        )

    st.success(" Analysis Complete")

    c1, c2, c3 = st.columns(3)
    c1.metric("Signal", result["signal"])
    c2.metric("Confidence", f"{result['confidence']}%")
    c3.metric("Price", f"${result['price']}")

    st.subheader("AI Analysis")
    st.markdown(ai_analysis)

    st.subheader("Price Chart (90 Days)")
    df = predictor.data.tail(90)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Close"))
    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Close"].rolling(50).mean(),
        name="SMA 50",
        line=dict(dash="dash")
    ))

    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Enter a stock ticker and click Analyze")

st.markdown("---")
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.caption(" This tool is for educational purposes only. Not financial advice.")
