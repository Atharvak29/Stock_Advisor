# Stock Price Prediction & Analysis App
# Streamlit Web Application

"""
INSTALLATION:
pip install streamlit yfinance pandas numpy scikit-learn plotly pickle5 requests
pip install xgboost lightgbm catboost

RUN APP:
streamlit run app.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json
from groq import Groq

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="AI Stock Predictor",
    page_icon="📈",
    layout="wide"
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def load_models():
    """Load pickled models and artifacts"""
    try:
        with open('models/preprocessing_artifacts.pkl', 'rb') as f:
            artifacts = pickle.load(f)
        
        with open('models/Gradient_Boosting.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('models/anomaly_OCSVM.pkl', 'rb') as f:
            anomaly_model = pickle.load(f)
            
        with open('models/anomaly_OCSVM_Scaler.pkl', 'rb') as f:
            anomaly_scaler = pickle.load(f)
        
        return model, artifacts, anomaly_model, anomaly_scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please run the Jupyter notebook first to train and save the models.")
        return None, None, None, None

def engineer_features(df):
    """Engineer features for prediction"""
    df = df.copy()
    
    # Basic features
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Price_Range'] = df['High'] - df['Low']
    df['Price_Change'] = df['Close'] - df['Open']
    
    # Moving Averages
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # Volume features
    df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA20']
    df['Volume_Change'] = df['Volume'].pct_change()
    
    # Volatility
    df['Volatility_10'] = df['Returns'].rolling(window=10).std() * np.sqrt(252)
    df['Volatility_30'] = df['Returns'].rolling(window=30).std() * np.sqrt(252)
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    # Momentum
    df['Momentum'] = df['Close'] - df['Close'].shift(10)
    df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
    
    # Stochastic
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['Stochastic_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    df['Stochastic_D'] = df['Stochastic_K'].rolling(window=3).mean()
    
    # OBV
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    # Price Position
    df['Price_Position'] = (df['Close'] - df['Low'].rolling(window=14).min()) / \
                           (df['High'].rolling(window=14).max() - df['Low'].rolling(window=14).min())
    
    # Lagged features
    for lag in [1, 2, 3, 5]:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
        df[f'Returns_Lag_{lag}'] = df['Returns'].shift(lag)
    
    return df

def query_groq(question, context=""):
    """Query Groq LLM via API"""
    api_key = st.session_state.get('groq_api_key', '')
    
    if not api_key:
        return "Please enter your Groq API key in the sidebar."
    
    try:
        client = Groq(api_key=api_key)
        
        messages = [
            {
                "role": "system",
                "content": "You are a financial expert assistant helping with stock market analysis and trading questions. Provide concise, accurate information."
            },
            {
                "role": "user",
                "content": f"{context}\n\nQuestion: {question}"
            }
        ]
        
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="openai/gpt-oss-20b",  # A fast and common model on Groq
            temperature=0.7,
            max_tokens=1024
        )
        
        return chat_completion.choices[0].message.content
    
    except Exception as e:
        return f"Error querying Groq: {str(e)}"

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.title("📈 AI Stock Price Predictor")
    st.markdown("**Predict stock movements & detect anomalies using ensemble ML models**")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Groq API Key
    groq_key = st.sidebar.text_input("Groq API Key", type="password", 
                                     help="Enter your Groq API key")
    if groq_key:
        st.session_state['groq_api_key'] = groq_key
    
    # Stock ticker input
    ticker = st.sidebar.text_input("Stock Ticker", value="AAPL",help="enter any yfianance ticker name eg : 'SPY','BTC-USD','TSLA'").upper()
    period = st.sidebar.selectbox("Time Period", 
                                   ["1mo", "3mo", "6mo", "1y", "2y","10y"],
                                   index=4)
    
    # Load models
    model, artifacts, anomaly_model, anomaly_scaler = load_models()
    
    if model is None:
        st.stop()
    
    # Fetch data button
    if st.sidebar.button("🔄 Fetch & Predict", type="primary"):
        with st.spinner(f"Fetching data for {ticker}..."):
            try:
                # Fetch stock data
                stock = yf.Ticker(ticker)
                df = stock.history(period=period)
                
                if df.empty:
                    st.error(f"No data found for {ticker}")
                    st.stop()
                
                # Store in session state
                st.session_state['df'] = df
                st.session_state['ticker'] = ticker
                
                # Engineer features
                df_features = engineer_features(df)
                st.session_state['df_features'] = df_features
                
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()
    
    # Display results if data exists
    if 'df' in st.session_state:
        df = st.session_state['df']
        df_features = st.session_state['df_features']
        ticker = st.session_state['ticker']
        
        # Get latest data for prediction
        latest_data = df_features[artifacts['feature_names']].iloc[-1:].values
        latest_scaled = artifacts['scaler'].transform(latest_data)
        
        # Make prediction
        prediction = model.predict(latest_scaled)[0]
        prediction_proba = model.predict_proba(latest_scaled)[0]
        predicted_class = artifacts['label_encoder'].inverse_transform([prediction])[0]
        
        # Display prediction
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
        
        with col2:
            color = "🟢" if predicted_class == "UP" else "🔴" if predicted_class == "DOWN" else "🟡"
            st.metric("Prediction", f"{color} {predicted_class}")
        
        with col3:
            confidence = prediction_proba[prediction] * 100
            st.metric("Confidence", f"{confidence:.1f}%")
        
        # Probability bars
        st.subheader("📊 Prediction Probabilities")
        prob_df = pd.DataFrame({
            'Movement': artifacts['label_encoder'].classes_,
            'Probability': prediction_proba * 100
        })
        
        fig_prob = go.Figure(data=[
            go.Bar(x=prob_df['Movement'], y=prob_df['Probability'],
                   marker_color=['red', 'yellow', 'green'])
        ])
        fig_prob.update_layout(
            yaxis_title="Probability (%)",
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig_prob, use_container_width=True)
        
        # Price Chart
        st.subheader("📈 Price Chart with Moving Averages")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], 
                                name='Close Price', line=dict(color='blue', width=2)))
        
        # Add MAs if available
        if 'MA20' in df_features.columns:
            fig.add_trace(go.Scatter(x=df_features.index, y=df_features['MA20'],
                                    name='MA20', line=dict(color='orange', width=1)))
        if 'MA50' in df_features.columns:
            fig.add_trace(go.Scatter(x=df_features.index, y=df_features['MA50'],
                                    name='MA50', line=dict(color='red', width=1)))
        
        fig.update_layout(height=400, xaxis_title="Date", yaxis_title="Price ($)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Anomaly Detection
        st.subheader("🔍 Anomaly Detection")
        
        anomaly_features = ['Volume', 'Volume_Ratio', 'Volume_Change', 
                           'Returns', 'Volatility_10', 'Volatility_30',
                           'Price_Range', 'ATR', 'RSI']
        
        available_features = [f for f in anomaly_features if f in df_features.columns]
        X_anomaly = df_features[available_features].fillna(df_features[available_features].mean())
        X_anomaly_scaled = anomaly_scaler.transform(X_anomaly)
        
        anomalies = anomaly_model.predict(X_anomaly_scaled)
        anomaly_labels = np.where(anomalies == -1, 1, 0)
        
        n_anomalies = np.sum(anomaly_labels)
        st.info(f"🚨 Detected {n_anomalies} anomalous trading days ({n_anomalies/len(anomalies)*100:.1f}%)")
        
        # Highlight anomalies on chart
        if n_anomalies > 0:
            anomaly_dates = df.index[anomaly_labels == 1]
            anomaly_prices = df.loc[anomaly_dates, 'Close']
            
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df.index, y=df['Close'],
                                     name='Price', mode='lines'))
            fig2.add_trace(go.Scatter(x=anomaly_dates, y=anomaly_prices,
                                     mode='markers', name='Anomalies',
                                     marker=dict(color='red', size=10, symbol='x')))
            fig2.update_layout(height=300, xaxis_title="Date", yaxis_title="Price ($)")
            st.plotly_chart(fig2, use_container_width=True)
        
        # Key Indicators
        st.subheader("📊 Key Technical Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            rsi_val = df_features['RSI'].iloc[-1] if 'RSI' in df_features.columns else 0
            st.metric("RSI", f"{rsi_val:.2f}")
        
        with col2:
            macd_val = df_features['MACD'].iloc[-1] if 'MACD' in df_features.columns else 0
            st.metric("MACD", f"{macd_val:.2f}")
        
        with col3:
            vol_ratio = df_features['Volume_Ratio'].iloc[-1] if 'Volume_Ratio' in df_features.columns else 0
            st.metric("Volume Ratio", f"{vol_ratio:.2f}")
        
        with col4:
            volatility = df_features['Volatility_10'].iloc[-1] if 'Volatility_10' in df_features.columns else 0
            st.metric("Volatility (10d)", f"{volatility:.2%}")
        
        # Groq AI Assistant
        st.subheader("🤖 Ask Groq AI about this Stock")
        
        if 'groq_api_key' in st.session_state and st.session_state['groq_api_key']:
            user_question = st.text_input("Ask a question about the stock or market:")
            
            if user_question:
                with st.spinner("Asking Groq..."):
                    context = f"""
                    Stock: {ticker}
                    Current Price: ${df['Close'].iloc[-1]:.2f}
                    Predicted Movement: {predicted_class}
                    RSI: {rsi_val:.2f}
                    Recent Performance: {df['Close'].pct_change().tail().mean()*100:.2f}% avg daily change
                    """
                    
                    response = query_groq(user_question, context)
                    st.markdown("**Groq's Response:**")
                    st.info(response)
        else:
            st.warning("⚠️ Please enter your Groq API key in the sidebar to use the AI assistant.")
        
        # Model Explanation
        with st.expander("ℹ️ How the Model Works"):
            st.markdown("""
            **Prediction Model:**
            - Uses a Stacking Ensemble combining Random Forest, Gradient Boosting, XGBoost, LightGBM, and CatBoost
            - Analyzes 50+ technical indicators including moving averages, RSI, MACD, volume patterns
            - Predicts next-day movement: UP (>0.5%), DOWN (<-0.5%), or STABLE
            
            **Anomaly Detection:**
            - One-Class SVM identifies unusual trading patterns
            - Detects volume spikes, volatility changes, and irregular price movements
            - Helps identify potential trading opportunities or risks
            
            **⚠️ Disclaimer:** This is for educational purposes only. Not financial advice.
            """)

if __name__ == "__main__":
    main()
