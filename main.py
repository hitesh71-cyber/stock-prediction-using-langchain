import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import os
import joblib
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# ----------------- CONFIG -----------------
st.set_page_config(
    page_title="Hybrid AI Financial Agent",
    layout="wide"
)

# ----------------- ENV -----------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    st.error("GOOGLE_API_KEY not found in environment variables.")
    st.stop()

# ----------------- LOAD MODEL -----------------
MODEL_FILE = "stock_xgb_model.joblib"

if not os.path.exists(MODEL_FILE):
    st.error("Trained model file not found.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

MODEL = load_model()

# ----------------- LLM -----------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.3,
    max_output_tokens=1024,
    google_api_key=GEMINI_API_KEY,
    safety_settings={
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
    }
)

# ----------------- STOCK PREDICTOR -----------------
class StockPredictor:
    def __init__(self, ticker):
        self.ticker = ticker
        self.model = MODEL
        self.data = None

    def fetch_data(self, period="2y"):
        stock = yf.Ticker(self.ticker)
        df = stock.history(period=period)
        if df.empty:
            return False
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        self.data = df
        return True

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

# ----------------- PROMPT -----------------
template = """
You are a senior financial analyst.

Stock: {ticker}
Price: {price}
Signal: {signal}
Confidence: {confidence}%
RSI: {rsi}
MACD: {macd}
SMA50: {sma50}

Write a short, simple explanation and add a disclaimer.
"""

prompt = PromptTemplate(
    input_variables=["ticker", "price", "signal", "confidence", "rsi", "macd", "sma50"],
    template=template
)

chain = prompt | llm

# ----------------- STREAMLIT UI -----------------
st.title("📈 Hybrid AI Financial Agent")
st.markdown("AI-powered stock analysis using ML + Gemini")

ticker = st.text_input(
    "Enter Stock Ticker",
    placeholder="AAPL, NVDA, RELIANCE.NS"
)

if st.button("Analyze"):
    if not ticker:
        st.warning("Please enter a ticker.")
        st.stop()

    ticker = ticker.upper()
    predictor = StockPredictor(ticker)

    with st.spinner("Fetching data & analyzing..."):
        if not predictor.fetch_data():
            st.error("Failed to fetch stock data.")
            st.stop()

        result = predictor.predict()

        response = chain.invoke({
            "ticker": ticker,
            "price": result["price"],
            "signal": result["signal"],
            "confidence": result["confidence"],
            "rsi": result["rsi"],
            "macd": result["macd"],
            "sma50": result["sma50"]
        })

    # ----------------- OUTPUT -----------------
    st.subheader(f"{result['signal']} | Confidence: {result['confidence']}%")
    st.markdown(response.content)

    # ----------------- CHART -----------------
    df = predictor.data.tail(90)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Price"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"].rolling(50).mean(), name="SMA 50"))

    fig.update_layout(
        height=450,
        template="plotly_white",
        title=f"{ticker} - 90 Day Chart"
    )

    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("⚠️ ML predictions are probabilistic. Not financial advice.")
