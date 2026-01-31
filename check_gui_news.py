"""
Quick script to check what news data the GUI would see
Run this to debug the 51 article issue
"""

import sys
import os
from ai_stock_predictor import AIStockPredictor

# Try to load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

print("=" * 80)
print("GUI NEWS DATA CHECK")
print("=" * 80)

symbol = "BTC-USD"
print(f"\nTesting with symbol: {symbol}")

# Initialize predictor (same as GUI would)
predictor = AIStockPredictor(symbol)

# Collect data with GUI-like parameters
print("\nCollecting data (same as GUI)...")
features = predictor.collect_data(
    period='6mo',
    interval='1h',
    news_days=30,  # Check if this is being set correctly
    max_news_per_source=1000
)

# Check news cache
news_cache = predictor.news_agent.news_cache
print(f"\nNews cache length: {len(news_cache)}")

# Check collection stats
stats = predictor.news_agent.collection_stats
print(f"\nCollection Statistics:")
print(f"  Sources used: {stats.get('sources_used', [])}")
print(f"  Sources failed: {stats.get('sources_failed', [])}")
print(f"  Total collected: {stats.get('total_collected', 0)}")
print(f"  Duplicates removed: {stats.get('duplicates_removed', 0)}")
print(f"  Final unique: {len(news_cache)}")

# Check dates
if news_cache:
    import pandas as pd
    from datetime import datetime
    news_df = pd.DataFrame(news_cache)
    if 'timestamp' in news_df.columns:
        news_df['date'] = pd.to_datetime(news_df['timestamp']).dt.date
        min_date = news_df['date'].min()
        max_date = news_df['date'].max()
        days_covered = (max_date - min_date).days + 1
        print(f"\nDate Analysis:")
        print(f"  Oldest article: {min_date}")
        print(f"  Newest article: {max_date}")
        print(f"  Days covered: {days_covered} days")
        
        # Check if all articles are from today
        today = datetime.now().date()
        today_count = len(news_df[news_df['date'] == today])
        print(f"  Articles from today: {today_count}")
        print(f"  Articles from other days: {len(news_df) - today_count}")
        
        if today_count == len(news_df):
            print("\n⚠️  WARNING: All articles are from today!")
            print("   This suggests date filtering is too restrictive or")
            print("   'News Lookback Days' might be set to 1 in the GUI")

print("\n" + "=" * 80)
