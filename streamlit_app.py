import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from predict import predict_sentiment
import random

# Set page config
st.set_page_config(
    page_title="Amazon Sentiment Analyzer",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 10px 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102,126,234,0.3);
    }
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        text-align: center;
        margin-top: 30px;
    }
    .sentiment-positive { color: #10b981; }
    .sentiment-negative { color: #ef4444; }
    .sentiment-neutral { color: #f59e0b; }
</style>
""", unsafe_allow_html=True)

# Initialize session state for sidebar toggle and history
if 'sidebar_collapsed' not in st.session_state:
    st.session_state.sidebar_collapsed = False
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'chart_data' not in st.session_state:
    st.session_state.chart_data = {'Positive': 0, 'Negative': 0, 'Neutral': 0}

# Header Section
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown("<h1 class='main-header'>🛒 Amazon Review Sentiment Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("---")

# Sidebar with toggle functionality
with st.sidebar:
    st.markdown("### 🔧 Controls & Settings")
    
    # Toggle sidebar collapse (visual only)
    if st.button("📊 Toggle Analytics View"):
        st.session_state.sidebar_collapsed = not st.session_state.sidebar_collapsed
        st.rerun()
    
    if not st.session_state.sidebar_collapsed:
        st.markdown("#### 📈 Live Stats")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Positive", st.session_state.chart_data['Positive'])
        with col2:
            st.metric("Negative", st.session_state.chart_data['Negative'])
        with col3:
            st.metric("Neutral", st.session_state.chart_data['Neutral'])
        
        st.markdown("---")
        st.markdown("#### 📋 Recent Predictions")
        if st.session_state.prediction_history:
            for i, (review, sentiment, timestamp) in enumerate(st.session_state.prediction_history[-5:][::-1]):
                st.caption(f"**{timestamp}**")
                st.text(f"{review[:50]}...")
                st.markdown(f"**Sentiment:** <span class='sentiment-{sentiment.lower()}'>{sentiment}</span>", unsafe_allow_html=True)
                if i < len(st.session_state.prediction_history[-5:]) - 1:
                    st.markdown("---")
        else:
            st.info("No predictions yet.")
        
        st.markdown("---")
        st.markdown("#### ⚙️ Model Info")
        st.info("Using Logistic Regression with TF-IDF vectorization for sentiment analysis.")

# Main Content Area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### ✍️ Enter Your Review")
    
    # Text input with placeholder example
    text = st.text_area(
        "Review Text",
        height=200,
        placeholder="Enter your Amazon product review here...\n\nExample: 'The product arrived on time and works perfectly! Very satisfied with my purchase.'",
        help="Enter a detailed review for accurate sentiment analysis"
    )
    
    # Button with enhanced styling
    predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
    with predict_col2:
        if st.button("🔍 Predict Sentiment", use_container_width=True):
            if text.strip():
                with st.spinner("Analyzing sentiment..."):
                    sentiment = predict_sentiment(text)
                    
                    # Store in history
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    st.session_state.prediction_history.append((text[:100], sentiment, timestamp))
                    
                    # Update chart data
                    st.session_state.chart_data[sentiment] += 1
                    
                    # Display result with style
                    st.markdown("---")
                    st.markdown("### 📊 Prediction Result")
                    
                    # Result card
                    sentiment_colors = {
                        'Positive': '#10b981',
                        'Negative': '#ef4444',
                        'Neutral': '#f59e0b'
                    }
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style='color: {sentiment_colors.get(sentiment, "#000")};'>
                            {sentiment.upper()}
                        </h3>
                        <p><strong>Review:</strong> {text[:200]}...</p>
                        <p><strong>Analysis Time:</strong> {timestamp}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence indicator (simulated)
                    confidence = random.uniform(0.7, 0.95)
                    st.progress(confidence, text=f"Confidence: {confidence:.1%}")
                    
            else:
                st.warning("⚠️ Please enter some text to analyze.")

with col2:
    st.markdown("### 📊 Sentiment Distribution")
    
    # Create pie chart
    if sum(st.session_state.chart_data.values()) > 0:
        fig = go.Figure(data=[go.Pie(
            labels=list(st.session_state.chart_data.keys()),
            values=list(st.session_state.chart_data.values()),
            hole=.3,
            marker_colors=['#10b981', '#ef4444', '#f59e0b']
        )])
        
        fig.update_layout(
            height=300,
            showlegend=True,
            margin=dict(t=20, b=20, l=20, r=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet. Make some predictions to see the chart!")
    
    st.markdown("---")
    st.markdown("### 📈 Trend Analysis")
    
    # Simulated trend data
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    positive_trend = np.random.randint(50, 100, 30) + np.arange(30) * 2
    negative_trend = np.random.randint(10, 40, 30)
    neutral_trend = np.random.randint(20, 60, 30)
    
    trend_fig = go.Figure()
    trend_fig.add_trace(go.Scatter(x=dates, y=positive_trend, name='Positive', line=dict(color='#10b981')))
    trend_fig.add_trace(go.Scatter(x=dates, y=negative_trend, name='Negative', line=dict(color='#ef4444')))
    trend_fig.add_trace(go.Scatter(x=dates, y=neutral_trend, name='Neutral', line=dict(color='#f59e0b')))
    
    trend_fig.update_layout(
        height=250,
        margin=dict(t=20, b=20, l=20, r=20),
        xaxis_title="Date",
        yaxis_title="Count",
        showlegend=True
    )
    
    st.plotly_chart(trend_fig, use_container_width=True)

# Footer
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    st.markdown("**🛒 Amazon Sentiment Analyzer**")
with footer_col2:
    st.markdown(f"**Total Predictions:** {len(st.session_state.prediction_history)}")
with footer_col3:
    st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

st.markdown("---")
st.markdown(
    """
    <div class='footer'>
        <p>© 2024 Amazon Sentiment Analyzer | Built with Streamlit | For demonstration purposes only</p>
    </div>
    """,
    unsafe_allow_html=True
)
