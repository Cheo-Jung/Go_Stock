"""
Diagnostic script to test news collection and see what's happening
Run this to debug why only 51 articles are being collected
"""

import os
import sys
from ai_stock_predictor import AIStockPredictor
import logging

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def test_news_collection():
    """Test news collection with detailed diagnostics"""
    
    print("=" * 80)
    print("NEWS COLLECTION DIAGNOSTIC TEST")
    print("=" * 80)
    
    # Check API keys
    print("\n1. Checking API Keys...")
    api_keys = {
        'NEWSAPI_KEY': os.getenv('NEWSAPI_KEY', ''),
        'ALPHAVANTAGE_API_KEY': os.getenv('ALPHAVANTAGE_API_KEY', ''),
        'FINNHUB_API_KEY': os.getenv('FINNHUB_API_KEY', ''),
        'POLYGON_API_KEY': os.getenv('POLYGON_API_KEY', ''),
        'TWITTER_BEARER_TOKEN': os.getenv('TWITTER_BEARER_TOKEN', ''),
        'REDDIT_CLIENT_ID': os.getenv('REDDIT_CLIENT_ID', ''),
        'NEWSCATCHER_API_KEY': os.getenv('NEWSCATCHER_API_KEY', ''),
        'BING_SEARCH_API_KEY': os.getenv('BING_SEARCH_API_KEY', ''),
        'CRYPTOCOMPARE_API_KEY': os.getenv('CRYPTOCOMPARE_API_KEY', ''),
    }
    
    for key_name, key_value in api_keys.items():
        status = "[FOUND]" if key_value else "[NOT FOUND]"
        print(f"  {key_name}: {status}")
        if key_value:
            print(f"    Value: {key_value[:10]}...{key_value[-5:] if len(key_value) > 15 else ''}")
    
    # Check .env file
    print("\n2. Checking .env file...")
    env_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'),
        os.path.join(os.getcwd(), '.env'),
    ]
    
    env_found = False
    for env_path in env_paths:
        if os.path.exists(env_path):
            print(f"  [OK] Found .env file at: {env_path}")
            env_found = True
            # Try to load it
            try:
                from dotenv import load_dotenv
                load_dotenv(env_path)
                print("  [OK] Successfully loaded .env file")
            except ImportError:
                print("  [WARN] python-dotenv not installed, trying manual load...")
                try:
                    with open(env_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#') and '=' in line:
                                key, value = line.split('=', 1)
                                os.environ[key.strip()] = value.strip()
                    print("  [OK] Manually loaded .env file")
                except Exception as e:
                    print(f"  [ERROR] Error loading .env: {e}")
            break
    
    if not env_found:
        print("  [ERROR] No .env file found!")
        print(f"    Checked paths: {env_paths}")
    
    # Test news collection
    print("\n3. Testing News Collection...")
    symbol = "BTC-USD"  # Test with Bitcoin
    print(f"  Symbol: {symbol}")
    
    try:
        predictor = AIStockPredictor(symbol)
        
        print("\n4. Collecting news with max_articles_per_source=1000...")
        news_data = predictor.news_agent.collect_news(
            days=30,
            sources=['all'],
            include_social=True,
            include_local=True,
            max_articles_per_source=1000
        )
        
        print(f"\n5. Collection Results:")
        print(f"  Total articles collected: {len(news_data)}")
        print(f"  Sources used: {predictor.news_agent.collection_stats.get('sources_used', [])}")
        print(f"  Sources failed: {predictor.news_agent.collection_stats.get('sources_failed', [])}")
        print(f"  Total before dedup: {predictor.news_agent.collection_stats.get('total_collected', 0)}")
        print(f"  Duplicates removed: {predictor.news_agent.collection_stats.get('duplicates_removed', 0)}")
        
        # Show articles by source
        if news_data:
            print(f"\n6. Articles by Source:")
            from collections import Counter
            sources = [item.get('source', 'Unknown') for item in news_data]
            source_counts = Counter(sources)
            for source, count in source_counts.most_common():
                print(f"  {source}: {count} articles")
        
        # Show first few articles
        print(f"\n7. Sample Articles (first 5):")
        for i, article in enumerate(news_data[:5], 1):
            print(f"\n  Article {i}:")
            print(f"    Title: {article.get('title', 'N/A')[:60]}...")
            print(f"    Source: {article.get('source', 'N/A')}")
            print(f"    Category: {article.get('category', 'N/A')}")
            print(f"    Timestamp: {article.get('timestamp', 'N/A')[:19]}")
        
        print("\n" + "=" * 80)
        print("DIAGNOSTIC COMPLETE")
        print("=" * 80)
        
        if len(news_data) <= 51:
            print("\n[WARNING] Only getting 51 articles!")
            print("\nPossible causes:")
            print("1. Only yfinance is working (others need API keys)")
            print("2. Other sources are failing silently")
            print("3. API keys not loaded from .env file")
            print("\nSolutions:")
            print("1. Check API keys are set in .env file")
            print("2. Verify .env file is in the project root")
            print("3. Restart your application after adding keys")
            print("4. Check the logs above for source failures")
        
    except Exception as e:
        print(f"\n[ERROR] Error during collection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_news_collection()
