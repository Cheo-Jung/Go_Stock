"""
Streamlit GUI for Advanced AI Stock Predictor System
Multi-Agent AI System with Ensemble Models
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import warnings
import sys
from pathlib import Path

# Import the AI Stock Predictor
from ai_stock_predictor import (
    AIStockPredictor, 
    TimeFrame, 
    MarketRegime,
    Prediction
)

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Stock Predictor - Advanced System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'predictor': None,
        'price_data': None,
        'features': None,
        'news_data': None,
        'models_trained': False,
        'prediction': None,
        'data_collected': False,
        'training_in_progress': False,
        'symbol': 'BTC-USD',
        'selected_timeframe': TimeFrame.MEDIUM_TERM,
        'model_performance': {},
        'prediction_history': [],
        'realtime_mode': False,
        'realtime_interval_seconds': 60,
        'realtime_market_interval': '1m',
        'last_realtime_refresh': None,
        'last_realtime_prediction_at': None,
        'last_realtime_anchor_time': None,
        'pending_realtime_predictions': [],
        'realtime_results': []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


def refresh_realtime_data() -> float:
    """Refresh market data and rebuild features using cached news for faster updates."""
    predictor = st.session_state.predictor
    price_data = predictor.market_agent.collect_price_data(
        period='5d',
        interval=st.session_state.realtime_market_interval
    )
    technical_data = predictor.technical_agent.calculate_indicators(price_data)
    macro_data = {
        'fear_greed': predictor.macro_agent.get_fear_greed_index(),
        'dxy': predictor.macro_agent.get_dxy()
    }
    news_data = predictor.news_agent.news_cache if predictor.news_agent.news_cache else []
    features = predictor.feature_engineer.create_features(price_data, news_data, technical_data, macro_data)

    predictor.price_data = price_data
    predictor.features = features
    st.session_state.price_data = price_data
    st.session_state.features = features
    return float(price_data['close'].iloc[-1])


def update_realtime_accuracy():
    """Resolve pending predictions when new actual prices become available."""
    if st.session_state.price_data is None or st.session_state.price_data.empty:
        return

    updated_pending = []
    for pending in st.session_state.pending_realtime_predictions:
        future_prices = st.session_state.price_data[
            st.session_state.price_data['datetime'] > pending['anchor_time']
        ]
        if future_prices.empty:
            updated_pending.append(pending)
            continue

        actual_price = float(future_prices['close'].iloc[0])
        absolute_error = abs(actual_price - pending['predicted_price'])
        pct_error = (absolute_error / actual_price) * 100 if actual_price else 0.0
        accuracy_pct = max(0.0, 100.0 - pct_error)

        st.session_state.realtime_results.append({
            'predicted_at': pending['predicted_at'],
            'resolved_at': datetime.now(),
            'predicted_price': pending['predicted_price'],
            'actual_price': actual_price,
            'absolute_error': absolute_error,
            'pct_error': pct_error,
            'accuracy_pct': accuracy_pct
        })

    st.session_state.pending_realtime_predictions = updated_pending

def schedule_realtime_refresh(refresh_ms: int = 5000):
    """Force browser-side refresh so continuous mode runs without manual clicks."""
    st.markdown(
        f"""
        <script>
            setTimeout(function() {{
                window.location.reload();
            }}, {refresh_ms});
        </script>
        """,
        unsafe_allow_html=True
    )


# Header
st.markdown('<div class="main-header">🤖 AI Stock Predictor - Advanced Multi-Agent System</div>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #666; margin-bottom: 2rem;'>
    Advanced ensemble-based prediction system with multiple specialized AI agents
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Asset type selection
    asset_type = st.radio("Asset Type", ["Cryptocurrency", "Stock"], index=0)
    
    if asset_type == "Cryptocurrency":
        symbol = st.text_input(
            "Symbol",
            value=st.session_state.symbol if st.session_state.symbol.endswith('-USD') else "BTC-USD",
            help="Example: BTC-USD, ETH-USD, XRP-USD"
        )
        popular_symbols = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", "SOL-USD"]
    else:
        symbol = st.text_input(
            "Symbol",
            value=st.session_state.symbol if not st.session_state.symbol.endswith('-USD') else "AAPL",
            help="Example: AAPL, TSLA, GOOGL, MSFT"
        )
        popular_symbols = ["AAPL", "TSLA", "GOOGL", "MSFT", "NVDA", "AMZN"]
    
    st.session_state.symbol = symbol
    
    # Quick symbol selection
    st.markdown("**Popular Symbols:**")
    cols = st.columns(3)
    for i, pop_symbol in enumerate(popular_symbols[:6]):
        if cols[i % 3].button(pop_symbol, key=f"btn_{pop_symbol}", use_container_width=True):
            st.session_state.symbol = pop_symbol
            st.rerun()
    
    st.divider()
    
    # Data collection settings
    st.subheader("📊 Data Collection")
    period = st.selectbox(
        "Period",
        options=["1mo", "3mo", "6mo", "1y", "2y"],
        index=2,
        help="Data collection period"
    )
    
    interval = st.selectbox(
        "Interval",
        options=["1h", "1d", "1wk"],
        index=1,
        help="Data collection interval"
    )
    
    st.divider()
    
    # News collection settings
    st.subheader("📰 News Collection")
    news_days = st.slider(
        "News Lookback Days",
        min_value=1,
        max_value=365,
        value=30,
        help="Number of days to look back for news"
    )
    
    max_news_per_source = st.slider(
        "Max Articles per Source",
        min_value=50,
        max_value=5000,
        value=1000,
        step=50,
        help="Maximum articles to collect from each source (higher = more articles but slower)"
    )
    
    st.info(f"💡 With {max_news_per_source} articles per source, you can collect up to {max_news_per_source * 10} articles from all sources combined")
    
    st.divider()
    
    # API Key Status Check
    st.subheader("🔑 API Key Status")
    with st.expander("Check API Keys", expanded=True):
        import os
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except:
            pass
        
        api_keys_status = {
            'NewsAPI': ('NEWSAPI_KEY', 'https://newsapi.org/register', '100/day free'),
            'Alpha Vantage': ('ALPHAVANTAGE_API_KEY', 'https://www.alphavantage.co/support/#api-key', '500/day free'),
            'Finnhub': ('FINNHUB_API_KEY', 'https://finnhub.io/register', '60/min free'),
            'Polygon.io': ('POLYGON_API_KEY', 'https://polygon.io/', 'Paid'),
            'Twitter/X': ('TWITTER_BEARER_TOKEN', 'https://developer.twitter.com/', 'Limited free'),
            'Reddit': ('REDDIT_CLIENT_ID', 'https://www.reddit.com/prefs/apps', 'Free'),
            'NewsCatcher': ('NEWSCATCHER_API_KEY', 'https://newscatcher.ai/', '100/month free'),
            'Bing News': ('BING_SEARCH_API_KEY', 'https://www.microsoft.com/en-us/bing/apis/bing-news-search-api', 'Azure account'),
        }
        
        # Check which keys are loaded
        keys_found = []
        keys_missing = []
        
        for service_name, (key_name, url, tier) in api_keys_status.items():
            key_value = os.getenv(key_name, '')
            if key_value:
                keys_found.append((service_name, key_name, url, tier))
            else:
                keys_missing.append((service_name, key_name, url, tier))
        
        # Show found keys
        if keys_found:
            st.success(f"✅ {len(keys_found)} API key(s) loaded:")
            for service_name, key_name, url, tier in keys_found:
                key_value = os.getenv(key_name, '')
                masked_key = key_value[:4] + "..." + key_value[-4:] if len(key_value) > 8 else "***"
                st.write(f"  • **{service_name}** ({tier}) - Key: `{masked_key}`")
        
        # Show missing keys
        if keys_missing:
            st.warning(f"⚠️ {len(keys_missing)} API key(s) missing:")
            for service_name, key_name, url, tier in keys_missing[:3]:  # Show first 3
                st.write(f"  • **{service_name}** - [Get key]({url}) ({tier})")
            if len(keys_missing) > 3:
                st.caption(f"  ... and {len(keys_missing) - 3} more")
        
        # Free sources (no API key needed)
        st.info("💡 **Free sources (no API key needed):**")
        st.write("  • yfinance (Yahoo Finance)")
        st.write("  • CryptoCompare (for crypto)")
        
        # Show status summary
        if keys_found:
            st.success(f"🎉 You have API keys! You can collect 200-4000+ articles.")
        else:
            st.warning("⚠️ No API keys found. You'll only get ~50 articles from free sources.")
            st.caption("💡 Add API keys to `.env` file to unlock more sources. See `QUICK_API_SETUP.md`")
    
    st.divider()
    
    # Model training settings
    st.subheader("🤖 Model Training")
    epochs = st.slider("Epochs", 10, 100, 30, 5)
    batch_size = st.slider("Batch Size", 8, 64, 32, 8)
    learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.001, 0.0001, format="%.4f")
    
    st.divider()

    # Real-time system settings
    st.subheader("⚡ Continuous Real-time")
    realtime_interval_seconds = st.slider(
        "Update Every (seconds)",
        min_value=10,
        max_value=300,
        value=st.session_state.realtime_interval_seconds,
        step=10,
        help="How often the app refreshes market data and emits a new prediction"
    )
    st.session_state.realtime_interval_seconds = realtime_interval_seconds

    realtime_market_interval = st.selectbox(
        "Market Data Granularity",
        options=["1m", "2m", "5m", "15m", "30m", "1h"],
        index=["1m", "2m", "5m", "15m", "30m", "1h"].index(st.session_state.realtime_market_interval)
        if st.session_state.realtime_market_interval in ["1m", "2m", "5m", "15m", "30m", "1h"] else 0,
        help="Granularity used during continuous mode"
    )
    st.session_state.realtime_market_interval = realtime_market_interval
    # Prediction settings
    st.subheader("🔮 Prediction")
    timeframe_options = {
        "Short-term (1-5 min)": TimeFrame.SHORT_TERM,
        "Medium-term (1h-1d)": TimeFrame.MEDIUM_TERM,
        "Long-term (1d-1w)": TimeFrame.LONG_TERM
    }
    
    selected_timeframe_name = st.selectbox(
        "Timeframe",
        options=list(timeframe_options.keys()),
        index=1
    )
    st.session_state.selected_timeframe = timeframe_options[selected_timeframe_name]
    
    st.divider()
    
    # System info
    st.subheader("ℹ️ System Info")
    if st.session_state.predictor:
        device = st.session_state.predictor.device
        st.info(f"Device: {device}")
        if st.session_state.models_trained:
            st.success("✅ Models Trained")
        if st.session_state.data_collected:
            st.success("✅ Data Collected")
    else:
        st.info("System not initialized")

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Dashboard",
    "📊 Data Collection",
    "🤖 Model Training",
    "🔮 Prediction",
    "📈 Analysis"
])

# ============================================================================
# TAB 1: DASHBOARD
# ============================================================================
with tab1:
    st.header("📊 Dashboard")
    
    if st.session_state.predictor is None:
        st.info("👈 Please initialize the system in the sidebar by collecting data first.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Symbol", st.session_state.symbol)
        
        with col2:
            if st.session_state.data_collected and st.session_state.price_data is not None:
                current_price = st.session_state.price_data['close'].iloc[-1]
                st.metric("Current Price", f"${current_price:,.2f}")
            else:
                st.metric("Current Price", "N/A")
        
        with col3:
            if st.session_state.prediction:
                pred_price = st.session_state.prediction.price
                st.metric("Predicted Price", f"${pred_price:,.2f}")
            else:
                st.metric("Predicted Price", "N/A")
        
        with col4:
            if st.session_state.prediction:
                confidence = st.session_state.prediction.confidence * 100
                st.metric("Confidence", f"{confidence:.1f}%")
            else:
                st.metric("Confidence", "N/A")
        
        st.divider()
        
        # Prediction details
        if st.session_state.prediction:
            pred = st.session_state.prediction
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Prediction Details")
                st.write(f"**Price:** ${pred.price:,.2f}")
                st.write(f"**Confidence:** {pred.confidence*100:.1f}%")
                st.write(f"**Price Range:** ${pred.lower_bound:,.2f} - ${pred.upper_bound:,.2f}")
                st.write(f"**Market Regime:** {pred.regime.value.upper()}")
                st.write(f"**Timeframe:** {pred.timeframe.value}")
                st.write(f"**Timestamp:** {pred.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            
            with col2:
                st.subheader("Model Contributions")
                contributions = pred.model_contributions
                for model_name, contribution in contributions.items():
                    st.progress(contribution, text=f"{model_name}: {contribution*100:.1f}%")
        
        # Price chart
        if st.session_state.price_data is not None:
            st.subheader("Price Chart")
            fig = go.Figure()
            
            price_data = st.session_state.price_data
            fig.add_trace(go.Scatter(
                x=price_data['datetime'] if 'datetime' in price_data.columns else price_data.index,
                y=price_data['close'],
                mode='lines',
                name='Close Price',
                line=dict(color='#1f77b4', width=2)
            ))
            
            if st.session_state.prediction:
                pred = st.session_state.prediction
                last_time = price_data['datetime'].iloc[-1] if 'datetime' in price_data.columns else price_data.index[-1]
                pred_time = last_time + timedelta(hours=1)
                
                fig.add_trace(go.Scatter(
                    x=[last_time, pred_time],
                    y=[price_data['close'].iloc[-1], pred.price],
                    mode='lines+markers',
                    name='Prediction',
                    line=dict(color='red', width=2, dash='dash'),
                    marker=dict(size=10)
                ))
                
                # Confidence interval
                fig.add_trace(go.Scatter(
                    x=[pred_time, pred_time],
                    y=[pred.lower_bound, pred.upper_bound],
                    mode='lines',
                    name='Confidence Interval',
                    line=dict(color='rgba(255,0,0,0.3)', width=8),
                    showlegend=False
                ))
            
            fig.update_layout(
                title=f"{st.session_state.symbol} Price Chart",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                height=500,
                hovermode='x unified',
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 2: DATA COLLECTION
# ============================================================================
with tab2:
    st.header("📊 Data Collection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Data Collection Process
        
        The system collects data from multiple specialized agents:
        - **Market Data Agent**: Real-time price data (OHLCV)
        - **News Sentiment Agent**: News articles and sentiment analysis
        - **Technical Analysis Agent**: 50+ technical indicators
        - **Macro Economic Agent**: Macroeconomic indicators
        - **Regime Detection Agent**: Market regime classification
        """)
    
    with col2:
        if st.button("🔄 Collect Data", type="primary", use_container_width=True):
            with st.spinner(f"Collecting data for {st.session_state.symbol}..."):
                try:
                    # Force re-initialize predictor if symbol changed
                    if (st.session_state.predictor is None or 
                        st.session_state.predictor.symbol != st.session_state.symbol):
                        st.session_state.predictor = AIStockPredictor(st.session_state.symbol)
                        st.session_state.news_data = None  # Clear old news data
                    
                    # Show collection parameters
                    st.write(f"**Collection Settings:**")
                    st.write(f"- Period: {period}, Interval: {interval}")
                    st.write(f"- News Days: {news_days}, Max per Source: {max_news_per_source}")
                    
                    # Collect data
                    features = st.session_state.predictor.collect_data(
                        period=period, 
                        interval=interval,
                        news_days=news_days,
                        max_news_per_source=max_news_per_source
                    )
                    
                    st.session_state.price_data = st.session_state.predictor.price_data
                    st.session_state.features = features
                    
                    # Get news data directly from the agent's cache - force refresh
                    news_cache = st.session_state.predictor.news_agent.news_cache
                    
                    # Debug: Log what we're getting
                    import logging
                    logging.info(f"DEBUG: news_cache length: {len(news_cache) if news_cache else 0}")
                    logging.info(f"DEBUG: collection_stats: {st.session_state.predictor.news_agent.collection_stats}")
                    
                    # Ensure we have the latest data
                    if news_cache:
                        st.session_state.news_data = news_cache.copy()
                    else:
                        # If cache is empty, try to get it from the last collection
                        # This shouldn't happen, but just in case
                        st.warning("⚠️ News cache is empty! Re-collecting...")
                        news_data = st.session_state.predictor.news_agent.collect_news(
                            days=news_days,
                            max_articles_per_source=max_news_per_source
                        )
                        st.session_state.news_data = news_data
                    
                    st.session_state.data_collected = True
                    
                    # Show collection statistics
                    stats = st.session_state.predictor.news_agent.collection_stats
                    
                    # Display detailed results
                    st.success(f"✅ Data collected successfully!")
                    
                    # Show news collection summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        article_count = len(st.session_state.news_data) if st.session_state.news_data else 0
                        st.metric("Total Articles", article_count)
                        # Debug display
                        if article_count <= 60:
                            st.caption(f"⚠️ Only free sources working")
                    with col2:
                        st.metric("Sources Used", len(stats.get('sources_used', [])))
                    with col3:
                        st.metric("Sources Failed", len(stats.get('sources_failed', [])))
                    
                    # Show which sources were used and how many articles each got
                    if stats.get('sources_used'):
                        st.info(f"📰 Sources attempted: {', '.join(stats.get('sources_used', []))}")
                        
                        # Count articles by source
                        if st.session_state.news_data:
                            from collections import Counter
                            sources = [item.get('source', 'Unknown') for item in st.session_state.news_data]
                            source_counts = Counter(sources)
                            
                            # Map source names to API key names
                            source_to_api = {
                                'newsapi': 'NEWSAPI_KEY',
                                'alphavantage': 'ALPHAVANTAGE_API_KEY',
                                'finnhub': 'FINNHUB_API_KEY',
                                'polygon': 'POLYGON_API_KEY',
                                'twitter': 'TWITTER_BEARER_TOKEN',
                                'reddit': 'REDDIT_CLIENT_ID',
                                'newscatcher': 'NEWSCATCHER_API_KEY',
                                'bing_news': 'BING_SEARCH_API_KEY',
                            }
                            
                            # Show which sources actually returned articles
                            st.write("**Articles by source:**")
                            source_cols = st.columns(min(5, len(source_counts)))
                            for idx, (source, count) in enumerate(source_counts.most_common(10)):
                                with source_cols[idx % len(source_cols)]:
                                    st.metric(source[:20], count)
                            
                            # Show API key usage status
                            st.divider()
                            st.write("**🔑 API Key Usage Status:**")
                            
                            import os
                            api_sources_working = []
                            api_sources_not_working = []
                            free_sources = []
                            
                            for source_name, count in source_counts.most_common():
                                source_lower = source_name.lower()
                                # Check if this is an API-key source
                                api_key_name = None
                                for key, api_key in source_to_api.items():
                                    if key in source_lower:
                                        api_key_name = api_key
                                        break
                                
                                if api_key_name:
                                    # This source requires an API key
                                    if os.getenv(api_key_name):
                                        api_sources_working.append((source_name, count, api_key_name))
                                    else:
                                        api_sources_not_working.append((source_name, count, api_key_name))
                                else:
                                    # Free source (yfinance, CryptoCompare, etc.)
                                    free_sources.append((source_name, count))
                            
                            # Display status
                            if api_sources_working:
                                st.success(f"✅ **API Keys Working** ({len(api_sources_working)} source(s)):")
                                for source_name, count, api_key in api_sources_working:
                                    st.write(f"  • {source_name}: {count} articles (using {api_key})")
                            
                            if api_sources_not_working:
                                st.warning(f"⚠️ **API Keys Missing** ({len(api_sources_not_working)} source(s) need keys):")
                                for source_name, count, api_key in api_sources_not_working:
                                    st.write(f"  • {source_name}: {count} articles (needs {api_key})")
                            
                            if free_sources:
                                st.info(f"💡 **Free Sources** ({len(free_sources)} source(s), no API key needed):")
                                for source_name, count in free_sources:
                                    st.write(f"  • {source_name}: {count} articles")
                    
                    # Check if API keys are missing
                    sources_used = stats.get('sources_used', [])
                    paid_sources = ['newsapi', 'alphavantage', 'finnhub', 'polygon', 'twitter', 'reddit', 'newscatcher', 'bing_news']
                    missing_keys = [s for s in paid_sources if s in sources_used]
                    
                    # Count articles from free vs paid sources
                    if st.session_state.news_data:
                        free_sources = ['yfinance', 'Yahoo Finance', 'CryptoCompare', 'cryptocompare']
                        free_count = sum(1 for item in st.session_state.news_data 
                                       if any(fs.lower() in item.get('source', '').lower() for fs in free_sources))
                        paid_count = article_count - free_count
                        
                        if paid_count == 0 and article_count <= 60:
                            st.warning("""
                            ⚠️ **Only free sources are working!**
                            
                            You're only getting articles from:
                            - ✅ yfinance (free, ~10-50 articles)
                            - ✅ CryptoCompare (free, ~50 articles)
                            
                            **To get 200-4000+ articles, add API keys:**
                            1. Create a `.env` file in your project folder
                            2. Get free API keys from:
                               - NewsAPI: https://newsapi.org/register (100/day free)
                               - Alpha Vantage: https://www.alphavantage.co/support/#api-key (500/day free)
                               - Finnhub: https://finnhub.io/register (60/min free)
                            3. Add to `.env`:
                               ```
                               NEWSAPI_KEY=your_key_here
                               ALPHAVANTAGE_API_KEY=your_key_here
                               FINNHUB_API_KEY=your_key_here
                               ```
                            4. Restart Streamlit and collect again
                            
                            See `QUICK_API_SETUP.md` for detailed instructions.
                            """)
                    
                    if stats.get('sources_failed'):
                        st.warning(f"⚠️ Sources failed (need API keys?): {', '.join(stats.get('sources_failed', []))}")
                    
                    # Show article breakdown by source
                    if st.session_state.news_data:
                        from collections import Counter
                        sources = [item.get('source', 'Unknown') for item in st.session_state.news_data]
                        source_counts = Counter(sources)
                        st.write("**Articles by source:**")
                        for source, count in source_counts.most_common(10):
                            st.write(f"- {source}: {count} articles")
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error collecting data: {str(e)}")
    
    st.divider()
    
    # Display collected data
    if st.session_state.data_collected:
        st.subheader("📈 Collected Data Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Price Data Points", len(st.session_state.price_data))
        
        with col2:
            news_count = len(st.session_state.news_data) if st.session_state.news_data else 0
            st.metric("News Articles", news_count)
            # Show if only yfinance articles (likely issue)
            if news_count <= 60 and st.session_state.predictor:
                stats = st.session_state.predictor.news_agent.collection_stats
                sources_used = stats.get('sources_used', [])
                if len(sources_used) <= 1:
                    st.caption("⚠️ Only 1 source used - add API keys for more!")
        
        with col3:
            if st.session_state.features is not None:
                st.metric("Features", len(st.session_state.features.columns))
            else:
                st.metric("Features", "N/A")
        
        with col4:
            if st.session_state.price_data is not None:
                date_range = f"{st.session_state.price_data['datetime'].iloc[0].strftime('%Y-%m-%d')} to {st.session_state.price_data['datetime'].iloc[-1].strftime('%Y-%m-%d')}"
                st.metric("Date Range", date_range[:15] + "...")
        
        # Show news collection diagnostics
        if st.session_state.predictor and hasattr(st.session_state.predictor.news_agent, 'collection_stats'):
            stats = st.session_state.predictor.news_agent.collection_stats
            with st.expander("🔍 News Collection Diagnostics", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Sources Used:**")
                    for source in stats.get('sources_used', []):
                        st.success(f"✓ {source}")
                with col2:
                    st.write("**Sources Failed:**")
                    failed = stats.get('sources_failed', [])
                    if failed:
                        for source in failed:
                            st.error(f"✗ {source}")
                    else:
                        st.info("None")
                
                st.write(f"**Total Collected (before dedup):** {stats.get('total_collected', 0)}")
                st.write(f"**Duplicates Removed:** {stats.get('duplicates_removed', 0)}")
                st.write(f"**Final Unique Articles:** {len(st.session_state.news_data) if st.session_state.news_data else 0}")
                
                # Show date range of articles
                if st.session_state.news_data:
                    try:
                            news_df = pd.DataFrame(st.session_state.news_data)
                            if 'timestamp' in news_df.columns:
                                try:
                                    dt_series = pd.to_datetime(news_df['timestamp'], format='ISO8601', errors='coerce')
                                    if dt_series.notna().any():
                                        news_df['date'] = dt_series.dt.date
                                        # Filter out NaT dates
                                        news_df = news_df[news_df['date'].notna()]
                                        if len(news_df) > 0:
                                            min_date = news_df['date'].min()
                                            max_date = news_df['date'].max()
                                            st.write(f"**Article Date Range:** {min_date} to {max_date}")
                                            st.write(f"**Days Covered:** {(max_date - min_date).days + 1} days")
                                except Exception:
                                    pass
                    except:
                        pass
        
        # Price data preview
        st.subheader("Price Data Preview")
        st.dataframe(
            st.session_state.price_data[['datetime', 'open', 'high', 'low', 'close', 'volume']].tail(10),
            use_container_width=True
        )
        
        # Features preview with detailed categorization
        if st.session_state.features is not None:
            st.subheader("📊 Feature Analysis")
            
            # Helper function to categorize features
            def categorize_features(feature_names):
                categories = {
                    'Base Price Features': [],
                    'Technical Indicators - Momentum': [],
                    'Technical Indicators - Trend': [],
                    'Technical Indicators - Volatility': [],
                    'Technical Indicators - Volume': [],
                    'Technical Indicators - Support/Resistance': [],
                    'Time-based Features': [],
                    'Lag Features': [],
                    'Rolling Statistics': [],
                    'News Sentiment Features': [],
                    'Macro Economic Features': [],
                    'Other Features': []
                }
                
                for feature in feature_names:
                    feature_lower = feature.lower()
                    
                    # Base price features
                    if feature in ['open', 'high', 'low', 'close', 'volume', 'vwap', 'returns', 'log_returns']:
                        categories['Base Price Features'].append(feature)
                    
                    # Momentum indicators
                    elif any(x in feature_lower for x in ['rsi', 'macd', 'stoch', 'momentum']):
                        categories['Technical Indicators - Momentum'].append(feature)
                    
                    # Trend indicators
                    elif any(x in feature_lower for x in ['sma_', 'ema_', 'trend', 'adx']):
                        categories['Technical Indicators - Trend'].append(feature)
                    
                    # Volatility indicators
                    elif any(x in feature_lower for x in ['atr', 'bb_', 'bollinger', 'volatility', 'std']):
                        categories['Technical Indicators - Volatility'].append(feature)
                    
                    # Volume indicators
                    elif any(x in feature_lower for x in ['obv', 'volume_', 'volume_sma', 'volume_ratio']):
                        categories['Technical Indicators - Volume'].append(feature)
                    
                    # Support/Resistance
                    elif any(x in feature_lower for x in ['support', 'resistance', 'price_position']):
                        categories['Technical Indicators - Support/Resistance'].append(feature)
                    
                    # Time-based features
                    elif any(x in feature_lower for x in ['hour', 'day_of_week', 'day_of_month', 'month', 
                                                          'dow_', 'is_weekend', 'is_month_end', 'sin', 'cos']):
                        categories['Time-based Features'].append(feature)
                    
                    # Lag features
                    elif 'lag_' in feature_lower:
                        categories['Lag Features'].append(feature)
                    
                    # Rolling statistics
                    elif 'rolling_' in feature_lower:
                        categories['Rolling Statistics'].append(feature)
                    
                    # News sentiment
                    elif 'news_' in feature_lower or 'sentiment' in feature_lower:
                        categories['News Sentiment Features'].append(feature)
                    
                    # Macro features
                    elif 'macro_' in feature_lower:
                        categories['Macro Economic Features'].append(feature)
                    
                    else:
                        categories['Other Features'].append(feature)
                
                return categories
            
            feature_cols = [col for col in st.session_state.features.columns 
                          if col not in ['datetime', 'open', 'high', 'low', 'close', 'volume']]
            
            categorized = categorize_features(feature_cols)
            
            # Display summary
            col1, col2, col3, col4 = st.columns(4)
            total_features = len(feature_cols)
            total_categories = sum(1 for cat_features in categorized.values() if cat_features)
            
            with col1:
                st.metric("Total Features", total_features)
            with col2:
                st.metric("Categories", total_categories)
            with col3:
                avg_per_category = total_features / total_categories if total_categories > 0 else 0
                st.metric("Avg per Category", f"{avg_per_category:.1f}")
            with col4:
                max_category = max(len(features) for features in categorized.values())
                st.metric("Max in Category", max_category)
            
            st.divider()
            
            # Display detailed categories
            st.markdown("### 📋 Feature Categories")
            
            # Create tabs for different category groups
            cat_tab1, cat_tab2, cat_tab3 = st.tabs(["📈 Technical Indicators", "⏰ Time & Lag Features", "📰 News & Macro"])
            
            with cat_tab1:
                # Technical Indicators
                tech_categories = [
                    ('Technical Indicators - Momentum', '🔄', '#FF6B6B'),
                    ('Technical Indicators - Trend', '📈', '#4ECDC4'),
                    ('Technical Indicators - Volatility', '📊', '#45B7D1'),
                    ('Technical Indicators - Volume', '📦', '#96CEB4'),
                    ('Technical Indicators - Support/Resistance', '🎯', '#FFEAA7'),
                ]
                
                for cat_name, icon, color in tech_categories:
                    features = categorized[cat_name]
                    if features:
                        with st.expander(f"{icon} {cat_name.replace('Technical Indicators - ', '')} ({len(features)} features)", expanded=False):
                            # Display in columns for better layout
                            cols = st.columns(min(3, len(features)))
                            for idx, feat in enumerate(sorted(features)):
                                cols[idx % 3].code(feat, language=None)
                            
                            # Show statistics if applicable
                            if features:
                                try:
                                    feat_data = st.session_state.features[features]
                                    st.markdown("**Statistics:**")
                                    stats_col1, stats_col2, stats_col3 = st.columns(3)
                                    with stats_col1:
                                        st.metric("Mean", f"{feat_data.mean().mean():.4f}")
                                    with stats_col2:
                                        st.metric("Std Dev", f"{feat_data.std().mean():.4f}")
                                    with stats_col3:
                                        st.metric("Non-Null", f"{feat_data.notna().sum().sum()}/{len(feat_data) * len(features)}")
                                except:
                                    pass
            
            with cat_tab2:
                # Time and Lag Features
                time_categories = [
                    ('Time-based Features', '⏰', '#A29BFE'),
                    ('Lag Features', '⏪', '#FD79A8'),
                    ('Rolling Statistics', '📉', '#FDCB6E'),
                ]
                
                for cat_name, icon, color in time_categories:
                    features = categorized[cat_name]
                    if features:
                        with st.expander(f"{icon} {cat_name} ({len(features)} features)", expanded=False):
                            # Group similar features
                            if 'Lag Features' in cat_name:
                                # Group by lag period
                                lag_groups = {}
                                for feat in sorted(features):
                                    if 'lag_' in feat:
                                        period = feat.split('lag_')[-1]
                                        if period not in lag_groups:
                                            lag_groups[period] = []
                                        lag_groups[period].append(feat)
                                
                                for period, group_features in sorted(lag_groups.items()):
                                    st.markdown(f"**Lag {period}:**")
                                    cols = st.columns(min(3, len(group_features)))
                                    for idx, feat in enumerate(group_features):
                                        cols[idx % 3].code(feat, language=None)
                            else:
                                cols = st.columns(min(3, len(features)))
                                for idx, feat in enumerate(sorted(features)):
                                    cols[idx % 3].code(feat, language=None)
                            
                            # Statistics
                            if features:
                                try:
                                    feat_data = st.session_state.features[features]
                                    st.markdown("**Statistics:**")
                                    stats_col1, stats_col2 = st.columns(2)
                                    with stats_col1:
                                        st.metric("Mean", f"{feat_data.mean().mean():.4f}")
                                    with stats_col2:
                                        st.metric("Std Dev", f"{feat_data.std().mean():.4f}")
                                except:
                                    pass
            
            with cat_tab3:
                # News and Macro
                other_categories = [
                    ('News Sentiment Features', '📰', '#6C5CE7'),
                    ('Macro Economic Features', '🌍', '#00B894'),
                    ('Base Price Features', '💰', '#FDCB6E'),
                    ('Other Features', '🔧', '#636E72'),
                ]
                
                for cat_name, icon, color in other_categories:
                    features = categorized[cat_name]
                    if features:
                        with st.expander(f"{icon} {cat_name} ({len(features)} features)", expanded=False):
                            cols = st.columns(min(3, len(features)))
                            for idx, feat in enumerate(sorted(features)):
                                cols[idx % 3].code(feat, language=None)
                            
                            # Statistics
                            if features:
                                try:
                                    feat_data = st.session_state.features[features]
                                    st.markdown("**Statistics:**")
                                    stats_col1, stats_col2, stats_col3 = st.columns(3)
                                    with stats_col1:
                                        st.metric("Mean", f"{feat_data.mean().mean():.4f}")
                                    with stats_col2:
                                        st.metric("Std Dev", f"{feat_data.std().mean():.4f}")
                                    with stats_col3:
                                        st.metric("Range", f"[{feat_data.min().min():.2f}, {feat_data.max().max():.2f}]")
                                except:
                                    pass
            
            # Feature distribution chart
            st.divider()
            st.markdown("### 📊 Feature Distribution by Category")
            
            category_counts = {k.replace('Technical Indicators - ', ''): len(v) 
                             for k, v in categorized.items() if v}
            
            if category_counts:
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(category_counts.keys()),
                        y=list(category_counts.values()),
                        marker_color='lightblue',
                        text=list(category_counts.values()),
                        textposition='auto'
                    )
                ])
                
                fig.update_layout(
                    title="Number of Features per Category",
                    xaxis_title="Category",
                    yaxis_title="Number of Features",
                    height=400,
                    template='plotly_white',
                    xaxis_tickangle=-45
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature search
            st.divider()
            st.markdown("### 🔍 Feature Search")
            search_term = st.text_input("Search for a feature", placeholder="e.g., rsi, macd, sma_20")
            
            if search_term:
                matching_features = [f for f in feature_cols if search_term.lower() in f.lower()]
                if matching_features:
                    st.success(f"Found {len(matching_features)} matching feature(s):")
                    for feat in matching_features:
                        # Find which category it belongs to
                        category = "Unknown"
                        for cat_name, features in categorized.items():
                            if feat in features:
                                category = cat_name.replace('Technical Indicators - ', '')
                                break
                        st.code(f"{feat} [{category}]", language=None)
                else:
                    st.warning(f"No features found matching '{search_term}'")
        
        # News data preview - Comprehensive view
        if st.session_state.news_data:
            st.subheader("📰 News Data Collection")
            
            # Force refresh news data from predictor if available
            if st.session_state.predictor and hasattr(st.session_state.predictor.news_agent, 'news_cache'):
                current_cache = st.session_state.predictor.news_agent.news_cache
                if current_cache and len(current_cache) > len(st.session_state.news_data):
                    st.info(f"🔄 Updating news data: Found {len(current_cache)} articles in cache (showing {len(st.session_state.news_data)})")
                    st.session_state.news_data = current_cache.copy()
                    st.rerun()
            
            news_df = pd.DataFrame(st.session_state.news_data)
            
            if not news_df.empty:
                # News statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total News", len(news_df))
                
                with col2:
                    unique_sources = news_df['source'].nunique() if 'source' in news_df.columns else 0
                    st.metric("Sources", unique_sources)
                
                with col3:
                    if 'category' in news_df.columns:
                        unique_categories = news_df['category'].nunique()
                        st.metric("Categories", unique_categories)
                    else:
                        st.metric("Categories", "N/A")
                
                with col4:
                    if 'sentiment' in news_df.columns:
                        avg_sentiment = news_df['sentiment'].mean()
                        st.metric("Avg Sentiment", f"{avg_sentiment:.2f}")
                    else:
                        st.metric("Avg Sentiment", "N/A")
                
                st.divider()
                
                # Organize news by different views
                news_tab1, news_tab2, news_tab3, news_tab4, news_tab5 = st.tabs([
                    "📋 All News",
                    "📊 By Source",
                    "🏷️ By Category",
                    "😊 By Sentiment",
                    "📈 Statistics"
                ])
                
                # Tab 1: All News
                with news_tab1:
                    st.markdown(f"### All News Articles ({len(news_df)} total)")
                    
                    # Search and filter
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        search_query = st.text_input("🔍 Search news", placeholder="Search by title or content...")
                    with col2:
                        date_filter = st.date_input(
                            "📅 Filter by date",
                            value=None,
                            help="Select date to filter news"
                        )
                    with col3:
                        sort_by = st.selectbox(
                            "Sort by",
                            options=['timestamp', 'sentiment', 'source', 'title'],
                            index=0
                        )
                    
                    # Apply filters
                    filtered_df = news_df.copy()
                    
                    if search_query:
                        mask = (
                            filtered_df['title'].str.contains(search_query, case=False, na=False) |
                            filtered_df.get('content', pd.Series([''] * len(filtered_df))).str.contains(search_query, case=False, na=False)
                        )
                        filtered_df = filtered_df[mask]
                    
                    if date_filter:
                        try:
                            dt_series = pd.to_datetime(filtered_df['timestamp'], format='ISO8601', errors='coerce')
                            if dt_series.notna().any():
                                filtered_df['date'] = dt_series.dt.date
                                filtered_df = filtered_df[filtered_df['date'] == date_filter]
                        except Exception:
                            pass
                    
                    # Sort
                    if sort_by in filtered_df.columns:
                        filtered_df = filtered_df.sort_values(by=sort_by, ascending=False)
                    
                    # Display news
                    st.markdown(f"**Showing {len(filtered_df)} of {len(news_df)} articles**")
                    
                    # Pagination
                    items_per_page = st.slider("Items per page", 10, 100, 25, 5)
                    total_pages = (len(filtered_df) + items_per_page - 1) // items_per_page
                    
                    if total_pages > 1:
                        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
                        start_idx = (page - 1) * items_per_page
                        end_idx = start_idx + items_per_page
                        page_df = filtered_df.iloc[start_idx:end_idx]
                    else:
                        page_df = filtered_df
                    
                    # Display news cards
                    for idx, row in page_df.iterrows():
                        with st.expander(f"📰 {row.get('title', 'No title')[:80]}...", expanded=False):
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.write(f"**Source:** {row.get('source', 'Unknown')}")
                                try:
                                    if row.get('timestamp'):
                                        dt = pd.to_datetime(row.get('timestamp'), format='ISO8601', errors='coerce')
                                        if pd.notna(dt):
                                            st.write(f"**Date:** {dt.strftime('%Y-%m-%d %H:%M:%S')}")
                                        else:
                                            st.write(f"**Date:** N/A")
                                    else:
                                        st.write(f"**Date:** N/A")
                                except:
                                    st.write(f"**Date:** {row.get('timestamp', 'N/A')}")
                                if 'category' in row:
                                    st.write(f"**Category:** {row['category']}")
                                
                                if 'content' in row and row['content']:
                                    st.write(f"**Content:** {row['content'][:500]}...")
                                elif 'title' in row:
                                    st.write(f"**Title:** {row['title']}")
                                
                                if 'url' in row and row['url']:
                                    st.markdown(f"[🔗 Read full article]({row['url']})")
                            
                            with col2:
                                if 'sentiment' in row:
                                    sentiment = row['sentiment']
                                    sentiment_color = 'green' if sentiment > 0 else 'red' if sentiment < 0 else 'gray'
                                    st.metric("Sentiment", f"{sentiment:.2f}", delta=f"{sentiment:.2f}")
                                    
                                    # Sentiment bar
                                    st.progress(abs(sentiment), text=f"{'Positive' if sentiment > 0 else 'Negative' if sentiment < 0 else 'Neutral'}")
                                
                                if 'engagement' in row and pd.notna(row['engagement']):
                                    st.metric("Engagement", int(row['engagement']))
                    
                    # Show pagination info
                    if total_pages > 1:
                        st.info(f"Page {page} of {total_pages} | Showing {start_idx+1}-{min(end_idx, len(filtered_df))} of {len(filtered_df)} articles")
                
                # Tab 2: By Source
                with news_tab2:
                    st.markdown("### News by Source")
                    
                    if 'source' in news_df.columns:
                        source_counts = news_df['source'].value_counts()
                        
                        # Source selection
                        selected_sources = st.multiselect(
                            "Select sources to view",
                            options=source_counts.index.tolist(),
                            default=source_counts.head(10).index.tolist()
                        )
                        
                        if selected_sources:
                            source_df = news_df[news_df['source'].isin(selected_sources)]
                            
                            # Source statistics
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Articles per Source:**")
                                fig = go.Figure(data=[
                                    go.Bar(
                                        x=source_counts[selected_sources].index,
                                        y=source_counts[selected_sources].values,
                                        marker_color='lightblue',
                                        text=source_counts[selected_sources].values,
                                        textposition='auto'
                                    )
                                ])
                                fig.update_layout(
                                    title="News Count by Source",
                                    xaxis_title="Source",
                                    yaxis_title="Number of Articles",
                                    height=400,
                                    template='plotly_white',
                                    xaxis_tickangle=-45
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                st.markdown("**Source Details:**")
                                for source in selected_sources:
                                    source_news = source_df[source_df['source'] == source]
                                    with st.expander(f"📰 {source} ({len(source_news)} articles)"):
                                        st.dataframe(
                                            source_news[['timestamp', 'title', 'sentiment', 'category']].head(20),
                                            use_container_width=True,
                                            hide_index=True
                                        )
                        else:
                            st.info("Select at least one source to view")
                    else:
                        st.warning("Source information not available")
                
                # Tab 3: By Category
                with news_tab3:
                    st.markdown("### News by Category")
                    
                    if 'category' in news_df.columns:
                        category_counts = news_df['category'].value_counts()
                        
                        # Category selection
                        selected_categories = st.multiselect(
                            "Select categories to view",
                            options=category_counts.index.tolist(),
                            default=category_counts.index.tolist()
                        )
                        
                        if selected_categories:
                            category_df = news_df[news_df['category'].isin(selected_categories)]
                            
                            # Category visualization
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                fig = go.Figure(data=[
                                    go.Pie(
                                        labels=category_counts.index,
                                        values=category_counts.values,
                                        hole=0.3
                                    )
                                ])
                                fig.update_layout(
                                    title="News Distribution by Category",
                                    height=400,
                                    template='plotly_white'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                st.markdown("**Category Summary:**")
                                for cat in selected_categories:
                                    cat_news = category_df[category_df['category'] == cat]
                                    st.metric(cat.capitalize(), len(cat_news))
                            
                            # News by category
                            for cat in selected_categories:
                                cat_news = category_df[category_df['category'] == cat]
                                with st.expander(f"🏷️ {cat.capitalize()} ({len(cat_news)} articles)", expanded=False):
                                    st.dataframe(
                                        cat_news[['timestamp', 'title', 'source', 'sentiment']].head(50),
                                        use_container_width=True,
                                        hide_index=True
                                    )
                        else:
                            st.info("Select at least one category to view")
                    else:
                        st.warning("Category information not available")
                
                # Tab 4: By Sentiment
                with news_tab4:
                    st.markdown("### News by Sentiment")
                    
                    if 'sentiment' in news_df.columns:
                        # Sentiment distribution
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Sentiment histogram
                            fig = go.Figure(data=[
                                go.Histogram(
                                    x=news_df['sentiment'],
                                    nbinsx=20,
                                    marker_color='steelblue'
                                )
                            ])
                            fig.update_layout(
                                title="Sentiment Distribution",
                                xaxis_title="Sentiment Score",
                                yaxis_title="Number of Articles",
                                height=400,
                                template='plotly_white'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Sentiment categories
                            positive = len(news_df[news_df['sentiment'] > 0.1])
                            negative = len(news_df[news_df['sentiment'] < -0.1])
                            neutral = len(news_df[(news_df['sentiment'] >= -0.1) & (news_df['sentiment'] <= 0.1)])
                            
                            st.metric("Positive", positive, delta=f"{(positive/len(news_df)*100):.1f}%")
                            st.metric("Neutral", neutral, delta=f"{(neutral/len(news_df)*100):.1f}%")
                            st.metric("Negative", negative, delta=f"{(negative/len(news_df)*100):.1f}%")
                        
                        # Filter by sentiment
                        sentiment_filter = st.selectbox(
                            "Filter by sentiment",
                            options=['All', 'Positive (>0.1)', 'Neutral (-0.1 to 0.1)', 'Negative (<-0.1)']
                        )
                        
                        if sentiment_filter == 'Positive (>0.1)':
                            filtered_sentiment = news_df[news_df['sentiment'] > 0.1]
                        elif sentiment_filter == 'Negative (<-0.1)':
                            filtered_sentiment = news_df[news_df['sentiment'] < -0.1]
                        elif sentiment_filter == 'Neutral (-0.1 to 0.1)':
                            filtered_sentiment = news_df[(news_df['sentiment'] >= -0.1) & (news_df['sentiment'] <= 0.1)]
                        else:
                            filtered_sentiment = news_df
                        
                        # Sort by sentiment
                        filtered_sentiment = filtered_sentiment.sort_values('sentiment', ascending=False)
                        
                        st.markdown(f"**Showing {len(filtered_sentiment)} articles**")
                        st.dataframe(
                            filtered_sentiment[['timestamp', 'title', 'source', 'sentiment']].head(100),
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.warning("Sentiment information not available")
                
                # Tab 5: Statistics
                with news_tab5:
                    st.markdown("### News Collection Statistics")
                    
                    # Collection stats from agent
                    if hasattr(st.session_state.predictor, 'news_agent') and hasattr(st.session_state.predictor.news_agent, 'collection_stats'):
                        stats = st.session_state.predictor.news_agent.collection_stats
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### Collection Overview")
                            st.metric("Total Collected", stats.get('total_collected', 0))
                            st.metric("Duplicates Removed", stats.get('duplicates_removed', 0))
                            st.metric("Unique Articles", len(news_df))
                        
                        with col2:
                            st.markdown("#### Sources Used")
                            sources_used = stats.get('sources_used', [])
                            for source in sources_used:
                                st.success(f"✅ {source}")
                            
                            st.markdown("#### Sources Failed")
                            sources_failed = stats.get('sources_failed', [])
                            for source in sources_failed:
                                st.error(f"❌ {source}")
                    
                    st.divider()
                    
                    # Time-based statistics
                    st.markdown("#### Time-based Analysis")
                    # Handle various timestamp formats (ISO8601, with/without microseconds, with/without timezone)
                    try:
                        # Convert to datetime first, then extract date and hour
                        dt_series = pd.to_datetime(news_df['timestamp'], format='ISO8601', errors='coerce')
                        # Only proceed if we have valid datetime values
                        if dt_series.notna().any():
                            news_df['date'] = dt_series.dt.date
                            news_df['hour'] = dt_series.dt.hour
                        else:
                            st.warning("⚠️ Could not parse timestamps for time-based analysis")
                            news_df['date'] = None
                            news_df['hour'] = None
                    except Exception as e:
                        st.warning(f"⚠️ Error parsing timestamps: {e}")
                        news_df['date'] = None
                        news_df['hour'] = None
                    
                    # Only show charts if we have valid date/hour data
                    if 'date' in news_df.columns and news_df['date'].notna().any():
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            daily_counts = news_df.groupby('date').size()
                            if len(daily_counts) > 0:
                                fig = go.Figure(data=[
                                    go.Scatter(
                                        x=daily_counts.index,
                                        y=daily_counts.values,
                                        mode='lines+markers',
                                        name='Articles per Day'
                                    )
                                ])
                                fig.update_layout(
                                    title="News Articles Over Time",
                                    xaxis_title="Date",
                                    yaxis_title="Number of Articles",
                                    height=300,
                                    template='plotly_white'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("No date data available for chart")
                        
                        with col2:
                            if 'hour' in news_df.columns and news_df['hour'].notna().any():
                                hourly_counts = news_df.groupby('hour').size()
                                if len(hourly_counts) > 0:
                                    fig = go.Figure(data=[
                                        go.Bar(
                                            x=hourly_counts.index,
                                            y=hourly_counts.values,
                                            marker_color='lightcoral'
                                        )
                                    ])
                                    fig.update_layout(
                                        title="News Articles by Hour of Day",
                                        xaxis_title="Hour",
                                        yaxis_title="Number of Articles",
                                        height=300,
                                        template='plotly_white'
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("No hour data available for chart")
                            else:
                                st.info("No hour data available for chart")
                    else:
                        st.info("⚠️ Timestamp data not available for time-based analysis")
                    
                    # Export option
                    st.divider()
                    st.markdown("#### Export News Data")
                    csv = news_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download News Data as CSV",
                        data=csv,
                        file_name=f"news_data_{st.session_state.symbol.replace('-', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            else:
                st.info("No news data available")
    else:
        st.info("👆 Click 'Collect Data' to start data collection")

# ============================================================================
# TAB 3: MODEL TRAINING
# ============================================================================
with tab3:
    st.header("🤖 Model Training & Loading")
    
    # Option 1: Load Pre-trained Model (Always available)
    st.subheader("📂 Load Pre-trained Model")
    st.markdown("""
    **Skip training by loading a pre-trained model file (.pt)**
    
    You can load a model that was previously trained and saved. This is faster than training from scratch.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload Model File (.pt)", 
            type=['pt'],
            help="Select a pre-trained model file to load"
        )
        
        # Also show existing model files in current directory
        try:
            import glob
            existing_models = glob.glob("*.pt")
            if existing_models:
                st.markdown("**Or select from existing models:**")
                selected_model = st.selectbox(
                    "Existing Model Files",
                    options=[""] + existing_models,
                    help="Select a model file from the current directory"
                )
            else:
                selected_model = None
        except:
            selected_model = None
    
    with col2:
        load_from_upload = st.button("📂 Load from Upload", type="primary", use_container_width=True, disabled=uploaded_file is None)
        load_from_file = st.button("📂 Load from File", type="primary", use_container_width=True, disabled=not selected_model or selected_model == "")
    
    if load_from_upload and uploaded_file is not None:
        with st.spinner("Loading model..."):
            try:
                # Save uploaded file temporarily
                with open("temp_model.pt", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Initialize predictor if needed
                if st.session_state.predictor is None:
                    st.session_state.predictor = AIStockPredictor(st.session_state.symbol)
                
                # Load the model
                st.session_state.predictor.load_system("temp_model.pt")
                st.session_state.models_trained = True
                
                # Try to restore price_data and features if available in the loaded model
                if hasattr(st.session_state.predictor, 'price_data') and st.session_state.predictor.price_data is not None:
                    st.session_state.price_data = st.session_state.predictor.price_data
                    st.session_state.data_collected = True
                if hasattr(st.session_state.predictor, 'features') and st.session_state.predictor.features is not None:
                    st.session_state.features = st.session_state.predictor.features
                
                st.success("✅ Model loaded successfully!")
                
                # Check if data is available
                if st.session_state.predictor.price_data is None or st.session_state.predictor.features is None:
                    st.warning("⚠️ **Data Required for Predictions**")
                    st.markdown("""
                    The model has been loaded, but you need to collect data to make predictions.
                    
                    **Next Steps:**
                    1. Go to the **📊 Data Collection** tab
                    2. Click **🔄 Collect Data** button
                    3. Once data is collected, you can make predictions
                    
                    The model will use the newly collected data for predictions.
                    """)
                else:
                    st.info("💡 Model and data are ready! You can now make predictions.")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error loading model: {str(e)}")
                st.exception(e)
    
    if load_from_file and selected_model:
        with st.spinner(f"Loading model from {selected_model}..."):
            try:
                # Initialize predictor if needed
                if st.session_state.predictor is None:
                    st.session_state.predictor = AIStockPredictor(st.session_state.symbol)
                
                # Load the model
                st.session_state.predictor.load_system(selected_model)
                st.session_state.models_trained = True
                
                # Try to restore price_data and features if available
                if hasattr(st.session_state.predictor, 'price_data') and st.session_state.predictor.price_data is not None:
                    st.session_state.price_data = st.session_state.predictor.price_data
                    st.session_state.data_collected = True
                if hasattr(st.session_state.predictor, 'features') and st.session_state.predictor.features is not None:
                    st.session_state.features = st.session_state.predictor.features
                
                st.success(f"✅ Model loaded successfully from {selected_model}!")
                
                # Check if data is available
                if st.session_state.predictor.price_data is None or st.session_state.predictor.features is None:
                    st.warning("⚠️ **Data Required for Predictions**")
                    st.markdown("""
                    The model has been loaded, but you need to collect data to make predictions.
                    
                    **Next Steps:**
                    1. Go to the **📊 Data Collection** tab
                    2. Click **🔄 Collect Data** button
                    3. Once data is collected, you can make predictions
                    
                    The model will use the newly collected data for predictions.
                    """)
                else:
                    st.info("💡 Model and data are ready! You can now make predictions.")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error loading model: {str(e)}")
                st.exception(e)
    
    st.divider()
    
    # Option 2: Train New Model (Only if data is collected)
    st.subheader("🚀 Train New Model")
    
    if not st.session_state.data_collected:
        st.warning("⚠️ Please collect data first in the 'Data Collection' tab before training!")
        st.info("💡 **Tip:** If you have a pre-trained model, you can load it above without needing to collect data first.")
    else:
        st.markdown("""
        ### Training Process
        
        The system trains three specialized models:
        1. **Short-term Model**: Transformer for 1-5 minute predictions
        2. **Medium-term Model**: LSTM + Transformer hybrid for 1 hour - 1 day
        3. **Long-term Model**: Transformer for 1 day - 1 week
        
        All models are trained on the collected features and combined in an ensemble.
        """)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Training Configuration")
            st.write(f"- **Epochs:** {epochs}")
            st.write(f"- **Batch Size:** {batch_size}")
            st.write(f"- **Learning Rate:** {learning_rate}")
            st.write(f"- **Features:** {len(st.session_state.features.columns) if st.session_state.features is not None else 'N/A'}")
        
        with col2:
            if st.button("🚀 Start Training", type="primary", use_container_width=True):
                if st.session_state.predictor is None:
                    st.error("Please collect data first!")
                else:
                    st.session_state.training_in_progress = True
                    
                    with st.spinner("Training models... This may take several minutes."):
                        try:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # Train models
                            st.session_state.predictor.train_models(
                                epochs=epochs,
                                batch_size=batch_size,
                                lr=learning_rate
                            )
                            
                            st.session_state.models_trained = True
                            st.session_state.training_in_progress = False
                            
                            progress_bar.progress(100)
                            status_text.success("✅ Training completed!")
                            
                            st.success("🎉 All models trained successfully!")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Training error: {str(e)}")
                            st.session_state.training_in_progress = False
        
        st.divider()
        
        # Training status
        if st.session_state.models_trained:
            st.subheader("✅ Model Status")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.success("✅ Short-term Model")
                st.info("Transformer architecture")
            
            with col2:
                st.success("✅ Medium-term Model")
                st.info("LSTM + Transformer hybrid")
            
            with col3:
                st.success("✅ Long-term Model")
                st.info("Transformer architecture")
            
            st.info("🎯 All models are ready for prediction!")
            
            # Model save
            st.subheader("💾 Save Model")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                model_name = st.text_input("Model Name", value=f"model_{st.session_state.symbol.replace('-', '_')}")
            
            with col2:
                if st.button("💾 Save Model", use_container_width=True):
                    try:
                        save_path = f"{model_name}.pt"
                        st.session_state.predictor.save_system(save_path)
                        st.success(f"✅ Model saved to {save_path}")
                        st.info(f"💡 You can load this model later using the 'Load Pre-trained Model' section above.")
                    except Exception as e:
                        st.error(f"❌ Error saving model: {str(e)}")

# ============================================================================
# TAB 4: PREDICTION
# ============================================================================
with tab4:
    st.header("🔮 Price Prediction")
    
    # Sync session state with predictor if needed
    if st.session_state.predictor is not None:
        if st.session_state.price_data is not None and st.session_state.predictor.price_data is None:
            st.session_state.predictor.price_data = st.session_state.price_data
        if st.session_state.features is not None and st.session_state.predictor.features is None:
            st.session_state.predictor.features = st.session_state.features
    
    # Status check
    status_ok = True
    status_messages = []
    
    if not st.session_state.models_trained:
        status_ok = False
        status_messages.append("❌ Model not trained/loaded")
    elif st.session_state.predictor is None:
        status_ok = False
        status_messages.append("❌ Predictor not initialized")
    elif st.session_state.predictor.price_data is None or st.session_state.predictor.features is None:
        status_ok = False
        status_messages.append("❌ Data not collected")
    elif len(st.session_state.predictor.features) < 60:
        status_ok = False
        status_messages.append(f"⚠️ Insufficient data ({len(st.session_state.predictor.features)}/60 points needed)")
    
    # Show status
    if status_ok:
        st.success("✅ Ready for predictions! Model and data are available.")
    else:
        st.error("**Status Check Failed:**")
        for msg in status_messages:
            st.write(f"  {msg}")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train or load a model first in the 'Model Training' tab!")
    elif st.session_state.predictor is None:
        st.error("❌ Predictor not initialized. Please collect data first.")
    elif st.session_state.predictor.price_data is None or st.session_state.predictor.features is None:
        st.warning("⚠️ **Data Required for Predictions**")
        st.markdown("""
        The model is loaded, but you need to collect data to make predictions.
        
        **To fix this:**
        1. Go to the **📊 Data Collection** tab
        2. Click **🔄 Collect Data** button
        3. Wait for data collection to complete
        4. Return here to make predictions
        
        The prediction requires:
        - Price data (at least 60 data points)
        - Feature data (calculated from price and news)
        """)
    elif len(st.session_state.predictor.features) < 60:
        st.warning(f"⚠️ **Insufficient Data**")
        st.markdown(f"""
        You have {len(st.session_state.predictor.features)} data points, but predictions require at least 60.
        
        **To fix this:**
        1. Go to the **📊 Data Collection** tab
        2. Increase the **Period** (e.g., from "3mo" to "6mo" or "1y")
        3. Click **🔄 Collect Data** again
        4. Return here to make predictions
        """)
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Prediction System
            
            The system uses an ensemble of specialized models to make predictions:
            - Models are weighted based on market regime
            - Confidence intervals are calculated
            - Feature importance is analyzed
            """)
        
        with col2:
            if st.button("🔮 Make Prediction", type="primary", use_container_width=True):
                with st.spinner("Generating prediction..."):
                    try:
                        prediction = st.session_state.predictor.predict(
                            timeframe=st.session_state.selected_timeframe
                        )
                        
                        st.session_state.prediction = prediction
                        st.session_state.prediction_history.append({
                            'timestamp': datetime.now(),
                            'prediction': prediction
                        })
                        
                        st.success("✅ Prediction generated!")
                        st.rerun()
                        
                    except ValueError as e:
                        error_msg = str(e)
                        if "Insufficient data" in error_msg or "data" in error_msg.lower():
                            st.error(f"❌ Prediction error: {error_msg}")
                            st.warning("""
                            **This usually means:**
                            - Not enough data points (need at least 60)
                            - Data hasn't been collected yet
                            
                            **Solution:**
                            1. Go to **📊 Data Collection** tab
                            2. Make sure you've collected data
                            3. If you have less than 60 data points, increase the **Period** setting
                            4. Click **🔄 Collect Data** again
                            5. Return here to make predictions
                            """)
                        else:
                            st.error(f"❌ Prediction error: {error_msg}")
                    except Exception as e:
                        st.error(f"❌ Prediction error: {str(e)}")
                        st.exception(e)

        st.subheader("⚡ Continuous Real-time Prediction")
        ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 2])
        with ctrl1:
            if st.button("▶️ Start Continuous", use_container_width=True):
                st.session_state.realtime_mode = True
                st.session_state.last_realtime_refresh = None
                st.session_state.last_realtime_anchor_time = None
                st.success("Continuous mode started")
        with ctrl2:
            if st.button("⏹️ Stop Continuous", use_container_width=True):
                st.session_state.realtime_mode = False
                st.info("Continuous mode stopped")
        with ctrl3:
            mode_status = "🟢 Running" if st.session_state.realtime_mode else "⚪ Idle"
            st.markdown(f"**Status:** {mode_status} | Interval: {st.session_state.realtime_interval_seconds}s")

        if st.session_state.realtime_mode:
            schedule_realtime_refresh(refresh_ms=5000)

            now = datetime.now()
            due = (
                st.session_state.last_realtime_refresh is None or
                (now - st.session_state.last_realtime_refresh).total_seconds() >= st.session_state.realtime_interval_seconds
            )
            if due:
                try:
                    refresh_realtime_data()
                    update_realtime_accuracy()

                    latest_anchor_time = st.session_state.price_data['datetime'].iloc[-1]
                    if st.session_state.last_realtime_anchor_time == latest_anchor_time:
                        st.info("No new market candle yet. Waiting for the next candle before scoring a new prediction.")
                    else:
                        realtime_prediction = st.session_state.predictor.predict(
                            timeframe=st.session_state.selected_timeframe
                        )
                        st.session_state.prediction = realtime_prediction
                        st.session_state.prediction_history.append({
                            'timestamp': now,
                            'prediction': realtime_prediction
                        })
                        st.session_state.pending_realtime_predictions.append({
                            'predicted_at': now,
                            'anchor_time': latest_anchor_time,
                            'predicted_price': realtime_prediction.price
                        })
                        st.session_state.last_realtime_prediction_at = now
                        st.session_state.last_realtime_anchor_time = latest_anchor_time
                        st.success(f"Continuous prediction updated at {now.strftime('%H:%M:%S')}")

                    st.session_state.last_realtime_refresh = now
                except Exception as e:
                    st.error(f"Continuous mode error: {e}")

            if st.session_state.realtime_results:
                realtime_df = pd.DataFrame(st.session_state.realtime_results[-20:])
                avg_acc = realtime_df['accuracy_pct'].mean()
                avg_err = realtime_df['pct_error'].mean()
                m1, m2, m3 = st.columns(3)
                m1.metric("Resolved Predictions", len(realtime_df))
                m2.metric("Average Accuracy", f"{avg_acc:.2f}%")
                m3.metric("Average % Error", f"{avg_err:.2f}%")
                chart_df = realtime_df.copy()
                chart_df['predicted_at'] = pd.to_datetime(chart_df['predicted_at'])
                st.line_chart(chart_df.set_index('predicted_at')[['accuracy_pct']])
                st.dataframe(
                    realtime_df[['predicted_at', 'predicted_price', 'actual_price', 'pct_error', 'accuracy_pct']].tail(10),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("Waiting for enough new candles to score prediction accuracy...")

        st.divider()
        
        # Display prediction
        if st.session_state.prediction:
            pred = st.session_state.prediction
            
            st.subheader("📊 Prediction Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Predicted Price", f"${pred.price:,.2f}")
            
            with col2:
                if st.session_state.price_data is not None:
                    current_price = st.session_state.price_data['close'].iloc[-1]
                    change = ((pred.price - current_price) / current_price) * 100
                    st.metric("Expected Change", f"{change:+.2f}%", delta=f"{change:+.2f}%")
            
            with col3:
                st.metric("Confidence", f"{pred.confidence*100:.1f}%")
            
            with col4:
                st.metric("Market Regime", pred.regime.value.upper())
            
            # Prediction details
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Price Range")
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=['Lower Bound', 'Prediction', 'Upper Bound'],
                    y=[pred.lower_bound, pred.price, pred.upper_bound],
                    marker_color=['red', 'blue', 'green'],
                    text=[f"${pred.lower_bound:,.2f}", f"${pred.price:,.2f}", f"${pred.upper_bound:,.2f}"],
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title="Prediction Range",
                    yaxis_title="Price (USD)",
                    height=400,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Model Contributions")
                contributions = pred.model_contributions
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(contributions.keys()),
                        y=[v*100 for v in contributions.values()],
                        marker_color='lightblue',
                        text=[f"{v*100:.1f}%" for v in contributions.values()],
                        textposition='auto'
                    )
                ])
                
                fig.update_layout(
                    title="Model Weight Contributions",
                    yaxis_title="Contribution (%)",
                    height=400,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Confidence explanation
            with st.expander("ℹ️ How Confidence is Calculated", expanded=False):
                st.markdown("""
                **Confidence Calculation:**
                
                Confidence is based on **model agreement** - how much the three models (short-term, medium-term, long-term) agree on their predictions.
                
                **Formula:**
                - Confidence = 1 / (1 + Coefficient of Variation)
                - Coefficient of Variation = Standard Deviation / Mean of model predictions
                
                **What this means:**
                - **High Confidence (60-100%)**: All models agree closely → reliable prediction
                - **Medium Confidence (30-60%)**: Models somewhat agree → moderate reliability
                - **Low Confidence (0-30%)**: Models disagree significantly → less reliable
                
                **Why confidence varies:**
                1. **Market volatility**: In volatile markets, models may disagree more
                2. **Data quality**: Better/more recent data → better agreement
                3. **Market regime**: Different regimes favor different models
                4. **Model training**: Well-trained models on similar data agree more
                
                **Tips:**
                - Use predictions with >50% confidence for trading decisions
                - Low confidence may indicate uncertain market conditions
                - Consider collecting more recent data if confidence is consistently low
                """)
                
                # Show individual model predictions if available
                if hasattr(st.session_state.predictor, 'ensemble'):
                    try:
                        # Get recent features for prediction
                        feature_cols = [col for col in st.session_state.predictor.features.columns 
                                       if col not in ['datetime', 'open', 'high', 'low', 'close', 'volume']]
                        recent_features = st.session_state.predictor.features[feature_cols].tail(60).values
                        
                        # Normalize
                        X_mean = recent_features.mean(axis=0)
                        X_std = recent_features.std(axis=0) + 1e-8
                        recent_features = (recent_features - X_mean) / X_std
                        
                        # Convert to tensor
                        import torch
                        X_tensor = torch.FloatTensor(recent_features).unsqueeze(0).to(st.session_state.predictor.device)
                        
                        # Get individual predictions
                        individual_preds = st.session_state.predictor.ensemble.predict(X_tensor, pred.regime)
                        
                        # Denormalize
                        current_price = st.session_state.price_data['close'].iloc[-1]
                        price_mean = st.session_state.price_data['close'].mean()
                        price_std = st.session_state.price_data['close'].std() + 1e-8
                        
                        st.markdown("**Individual Model Predictions:**")
                        model_prices = {}
                        for name, pred_tensor in individual_preds.items():
                            denorm_price = pred_tensor.item() * price_std + price_mean
                            model_prices[name] = denorm_price
                            change = ((denorm_price - current_price) / current_price) * 100
                            st.write(f"  - **{name.replace('_', ' ').title()}**: ${denorm_price:,.2f} ({change:+.2f}%)")
                        
                        # Calculate agreement
                        import numpy as np
                        pred_values = list(model_prices.values())
                        std_dev = np.std(pred_values)
                        mean_price = np.mean(pred_values)
                        cv = std_dev / (mean_price + 1e-8) if mean_price > 0 else 0
                        
                        st.markdown(f"""
                        **Agreement Metrics:**
                        - Standard Deviation: ${std_dev:,.2f}
                        - Coefficient of Variation: {cv:.4f}
                        - Price Range: ${min(pred_values):,.2f} - ${max(pred_values):,.2f}
                        """)
                    except Exception as e:
                        st.caption(f"Could not display individual predictions: {e}")
            
            # Feature importance
            with st.expander("📊 Feature Importance (Top 10)"):
                importance = pred.feature_importance
                sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=[v*100 for k, v in sorted_importance],
                        y=[k for k, v in sorted_importance],
                        orientation='h',
                        marker_color='steelblue'
                    )
                ])
                
                fig.update_layout(
                    title="Top 10 Most Important Features",
                    xaxis_title="Importance (%)",
                    height=400,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👆 Click 'Make Prediction' to generate a price prediction")

# ============================================================================
# TAB 5: ANALYSIS
# ============================================================================
with tab5:
    st.header("📈 Analysis & Insights")
    
    if not st.session_state.data_collected:
        st.warning("⚠️ Please collect data first!")
    else:
        # Technical indicators visualization
        st.subheader("📊 Technical Indicators")
        
        if st.session_state.price_data is not None:
            price_data = st.session_state.price_data.copy()
            
            # Calculate some basic indicators for visualization
            price_data['sma_20'] = price_data['close'].rolling(20).mean()
            price_data['sma_50'] = price_data['close'].rolling(50).mean()
            
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=('Price with Moving Averages', 'Volume'),
                row_heights=[0.7, 0.3]
            )
            
            # Price and moving averages
            fig.add_trace(
                go.Scatter(x=price_data['datetime'], y=price_data['close'], name='Close', line=dict(color='blue')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=price_data['datetime'], y=price_data['sma_20'], name='SMA 20', line=dict(color='orange')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=price_data['datetime'], y=price_data['sma_50'], name='SMA 50', line=dict(color='red')),
                row=1, col=1
            )
            
            # Volume
            fig.add_trace(
                go.Bar(x=price_data['datetime'], y=price_data['volume'], name='Volume', marker_color='lightblue'),
                row=2, col=1
            )
            
            fig.update_layout(height=600, title_text=f"{st.session_state.symbol} Technical Analysis", template='plotly_white')
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Market regime analysis
        if st.session_state.predictor and st.session_state.price_data is not None:
            st.subheader("📊 Market Regime Analysis")
            
            try:
                regime = st.session_state.predictor.regime_agent.detect_regime(st.session_state.price_data)
                
                regime_colors = {
                    MarketRegime.BULL: '🟢',
                    MarketRegime.BEAR: '🔴',
                    MarketRegime.SIDEWAYS: '🟡',
                    MarketRegime.VOLATILE: '🟠',
                    MarketRegime.UNKNOWN: '⚪'
                }
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Current Regime", f"{regime_colors.get(regime, '⚪')} {regime.value.upper()}")
                
                with col2:
                    if st.session_state.price_data is not None:
                        recent_returns = st.session_state.price_data['close'].pct_change().tail(20).mean() * 100
                        st.metric("Recent Returns (20 periods)", f"{recent_returns:.2f}%")
                
                with col3:
                    if st.session_state.price_data is not None:
                        volatility = st.session_state.price_data['close'].pct_change().tail(20).std() * 100
                        st.metric("Volatility (20 periods)", f"{volatility:.2f}%")
                
            except Exception as e:
                st.warning(f"Could not analyze regime: {str(e)}")
        
        # Prediction history
        if st.session_state.prediction_history:
            st.subheader("📜 Prediction History")
            
            history_df = pd.DataFrame([
                {
                    'Timestamp': entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                    'Predicted Price': f"${entry['prediction'].price:,.2f}",
                    'Confidence': f"{entry['prediction'].confidence*100:.1f}%",
                    'Regime': entry['prediction'].regime.value.upper()
                }
                for entry in st.session_state.prediction_history[-10:]
            ])
            
            st.dataframe(history_df, use_container_width=True, hide_index=True)
        
        # Statistics
        if st.session_state.price_data is not None:
            st.subheader("📊 Statistical Summary")
            
            price_stats = st.session_state.price_data['close'].describe()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Price Statistics:**")
                st.dataframe(price_stats, use_container_width=True)
            
            with col2:
                if st.session_state.prediction:
                    st.write("**Prediction Statistics:**")
                    st.metric("Predicted Price", f"${st.session_state.prediction.price:,.2f}")
                    st.metric("Confidence", f"{st.session_state.prediction.confidence*100:.1f}%")
                    st.metric("Price Range", 
                             f"${st.session_state.prediction.lower_bound:,.2f} - ${st.session_state.prediction.upper_bound:,.2f}")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <small>AI Stock Predictor - Advanced Multi-Agent System | Built with Streamlit & PyTorch</small>
</div>
""", unsafe_allow_html=True)
