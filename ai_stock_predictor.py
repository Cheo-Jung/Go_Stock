"""
Advanced Multi-Agent AI System for Stock/Crypto Price Prediction
Implements a comprehensive ensemble-based prediction system with multiple specialized agents
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
import yfinance as yf
import requests
import json
import os
import warnings
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque
import pickle
from pathlib import Path
import copy

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class MarketRegime(Enum):
    """Market regime types"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


class TimeFrame(Enum):
    """Prediction timeframes"""
    SHORT_TERM = "short"  # 1-5 minutes
    MEDIUM_TERM = "medium"  # 1 hour - 1 day
    LONG_TERM = "long"  # 1 day - 1 week


@dataclass
class Prediction:
    """Prediction result with confidence"""
    price: float
    confidence: float
    upper_bound: float
    lower_bound: float
    regime: MarketRegime
    timeframe: TimeFrame
    timestamp: datetime
    feature_importance: Dict[str, float]
    model_contributions: Dict[str, float]


@dataclass
class MarketData:
    """Market data structure"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None


# ============================================================================
# DATA COLLECTION AGENTS
# ============================================================================

class MarketDataAgent:
    """Agent for collecting real-time market data"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.data_cache = deque(maxlen=10000)
        self.last_data_source = 'unknown'
        

    def _generate_fallback_price_data(self, period: str = '1y', interval: str = '1h') -> pd.DataFrame:
        """Generate synthetic OHLCV data if external market API is unavailable."""
        now = datetime.now()
        period_points = {
            '5d': 5, '1mo': 30, '3mo': 90, '6mo': 180, '1y': 365, '2y': 730
        }
        interval_minutes = {
            '1m': 1, '2m': 2, '5m': 5, '15m': 15, '30m': 30, '1h': 60,
            '1d': 60 * 24, '1wk': 60 * 24 * 7
        }

        minutes = interval_minutes.get(interval, 60)
        days = period_points.get(period, 365)
        # Keep enough rows for long rolling windows (up to 200) and sequence creation.
        points = max(600, int((days * 24 * 60) / minutes))

        time_index = pd.date_range(end=now, periods=points, freq=f'{minutes}min')

        # Deterministic synthetic path based on symbol hash so runs are stable-ish.
        seed = abs(hash(f'{self.symbol}-{period}-{interval}')) % (2 ** 32)
        rng = np.random.default_rng(seed)

        base_price = 50000.0 if '-USD' in self.symbol else 150.0
        base_volatility = 0.0025 if minutes <= 60 else 0.01
        # Build regime + autocorrelated returns so fallback resembles tradable structure.
        regime_len = max(30, points // 12)
        regime_drifts = []
        while len(regime_drifts) * regime_len < points:
            regime_drifts.append(rng.choice([-0.00025, -0.0001, 0.0001, 0.00025]))
        drift_series = np.repeat(regime_drifts, regime_len)[:points]

        returns = np.zeros(points)
        noise = rng.normal(0.0, base_volatility, size=points)
        phi = 0.55
        for i in range(1, points):
            returns[i] = drift_series[i] + phi * returns[i - 1] + noise[i]
        close = base_price * np.exp(np.cumsum(returns))

        open_price = np.roll(close, 1)
        open_price[0] = close[0] * (1 - rng.normal(0, 0.001))
        spread = np.abs(rng.normal(0.0015, 0.0008, size=points))
        high = np.maximum(open_price, close) * (1 + spread)
        low = np.minimum(open_price, close) * (1 - spread)
        volume = rng.lognormal(mean=10.0, sigma=0.4, size=points)

        data = pd.DataFrame({
            'datetime': time_index,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
        data['vwap'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
        self.last_data_source = 'synthetic'
        logger.warning('Using synthetic fallback market data because live fetch failed')
        return data

    def collect_price_data(self, period: str = '1y', interval: str = '1h') -> pd.DataFrame:
        """Collect price data from yfinance"""
        logger.info(f"Collecting market data for {self.symbol}")
        try:
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for {self.symbol}")
            
            data.reset_index(inplace=True)
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            
            # Ensure datetime column exists (handle different yfinance versions)
            if 'date' in data.columns:
                data.rename(columns={'date': 'datetime'}, inplace=True)
            elif 'datetime' not in data.columns:
                # If index was datetime, it should be in columns after reset_index
                # If not, create from index
                if isinstance(data.index, pd.DatetimeIndex):
                    data.insert(0, 'datetime', data.index)
                else:
                    # Try to infer from first column or create sequential dates
                    data.insert(0, 'datetime', pd.date_range(start='2020-01-01', periods=len(data), freq='1H'))
            
            # Ensure datetime is datetime type
            data['datetime'] = pd.to_datetime(data['datetime'])
            
            # Calculate VWAP
            if 'volume' in data.columns and 'close' in data.columns:
                data['vwap'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
            
            self.last_data_source = 'live'
            logger.info(f"Collected {len(data)} data points")
            return data
            
        except Exception as e:
            logger.error(f"Error collecting market data: {e}")
            return self._generate_fallback_price_data(period=period, interval=interval)
    
    def get_order_book_snapshot(self) -> Dict:
        """Get order book snapshot (placeholder for real implementation)"""
        # In production, this would connect to exchange APIs
        return {
            'bids': [],
            'asks': [],
            'timestamp': datetime.now()
        }


class NewsSentimentAgent:
    """
    Comprehensive News Collection Agent
    Collects news from multiple sources: economic, social, local, and financial
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.news_cache = []
        self.collection_stats = {
            'total_collected': 0,
            'sources_used': [],
            'sources_failed': [],
            'duplicates_removed': 0
        }
        
    def collect_news(self, days: int = 30, sources: List[str] = None, 
                    include_social: bool = True, include_local: bool = True,
                    max_articles_per_source: int = 1000) -> List[Dict]:
        """
        Collect news from multiple comprehensive sources
        
        Args:
            days: Number of days to look back
            sources: List of specific sources to use (None = all available)
            include_social: Whether to include social media sources
            include_local: Whether to include local/regional news
            max_articles_per_source: Maximum articles to fetch per source (default: 1000)
        """
        if sources is None:
            sources = ['all']  # Collect from all available sources
        
        all_news = []
        self.collection_stats = {
            'total_collected': 0,
            'sources_used': [],
            'sources_failed': [],
            'duplicates_removed': 0
        }
        
        # ========================================================================
        # ECONOMIC & FINANCIAL NEWS SOURCES
        # ========================================================================
        logger.info(f"Starting news collection from sources: {sources}")
        logger.info(f"Max articles per source: {max_articles_per_source}, Days: {days}")
        
        if 'all' in sources or 'yfinance' in sources:
            try:
                logger.info("Attempting to fetch from yfinance...")
                news = self._fetch_yfinance_news(days=days, max_articles=max_articles_per_source)
                all_news.extend(news)
                self.collection_stats['sources_used'].append('yfinance')
                logger.info(f"✓ Collected {len(news)} articles from yfinance")
            except Exception as e:
                logger.warning(f"✗ Failed to fetch yfinance news: {e}")
                self.collection_stats['sources_failed'].append('yfinance')
        
        if 'all' in sources or 'newsapi' in sources:
            try:
                logger.info("Attempting to fetch from NewsAPI...")
                news = self._fetch_newsapi_news(days, max_articles=max_articles_per_source)
                all_news.extend(news)
                self.collection_stats['sources_used'].append('newsapi')
                logger.info(f"✓ Collected {len(news)} articles from NewsAPI")
            except Exception as e:
                logger.warning(f"✗ Failed to fetch NewsAPI news: {e}")
                self.collection_stats['sources_failed'].append('newsapi')
        
        if 'all' in sources or 'alphavantage' in sources:
            try:
                logger.info("Attempting to fetch from Alpha Vantage...")
                news = self._fetch_alphavantage_news(days, max_articles=max_articles_per_source)
                all_news.extend(news)
                self.collection_stats['sources_used'].append('alphavantage')
                logger.info(f"✓ Collected {len(news)} articles from Alpha Vantage")
            except Exception as e:
                logger.warning(f"✗ Failed to fetch Alpha Vantage news: {e}")
                self.collection_stats['sources_failed'].append('alphavantage')
        
        if 'all' in sources or 'finnhub' in sources:
            try:
                logger.info("Attempting to fetch from Finnhub...")
                news = self._fetch_finnhub_news(days, max_articles=max_articles_per_source)
                all_news.extend(news)
                self.collection_stats['sources_used'].append('finnhub')
                logger.info(f"✓ Collected {len(news)} articles from Finnhub")
            except Exception as e:
                logger.warning(f"✗ Failed to fetch Finnhub news: {e}")
                self.collection_stats['sources_failed'].append('finnhub')
        
        if 'all' in sources or 'polygon' in sources:
            try:
                logger.info("Attempting to fetch from Polygon...")
                news = self._fetch_polygon_news(days)
                all_news.extend(news)
                self.collection_stats['sources_used'].append('polygon')
                logger.info(f"✓ Collected {len(news)} articles from Polygon")
            except Exception as e:
                logger.warning(f"✗ Failed to fetch Polygon news: {e}")
                self.collection_stats['sources_failed'].append('polygon')
        
        # ========================================================================
        # SOCIAL MEDIA SOURCES
        # ========================================================================
        if include_social:
            if 'all' in sources or 'twitter' in sources:
                try:
                    logger.info("Attempting to fetch from Twitter...")
                    news = self._fetch_twitter_news(days)
                    all_news.extend(news)
                    self.collection_stats['sources_used'].append('twitter')
                    logger.info(f"✓ Collected {len(news)} tweets from Twitter")
                except Exception as e:
                    logger.warning(f"✗ Failed to fetch Twitter news: {e}")
                    self.collection_stats['sources_failed'].append('twitter')
            
            if 'all' in sources or 'reddit' in sources:
                try:
                    logger.info("Attempting to fetch from Reddit...")
                    news = self._fetch_reddit_news(days)
                    all_news.extend(news)
                    self.collection_stats['sources_used'].append('reddit')
                    logger.info(f"✓ Collected {len(news)} posts from Reddit")
                except Exception as e:
                    logger.warning(f"✗ Failed to fetch Reddit news: {e}")
                    self.collection_stats['sources_failed'].append('reddit')
        
        # ========================================================================
        # LOCAL & REGIONAL NEWS
        # ========================================================================
        if include_local:
            if 'all' in sources or 'newscatcher' in sources:
                try:
                    logger.info("Attempting to fetch from NewsCatcher...")
                    news = self._fetch_newscatcher_news(days)
                    all_news.extend(news)
                    self.collection_stats['sources_used'].append('newscatcher')
                    logger.info(f"✓ Collected {len(news)} articles from NewsCatcher")
                except Exception as e:
                    logger.warning(f"✗ Failed to fetch NewsCatcher news: {e}")
                    self.collection_stats['sources_failed'].append('newscatcher')
            
            if 'all' in sources or 'bing_news' in sources:
                try:
                    logger.info("Attempting to fetch from Bing News...")
                    news = self._fetch_bing_news(days)
                    all_news.extend(news)
                    self.collection_stats['sources_used'].append('bing_news')
                    logger.info(f"✓ Collected {len(news)} articles from Bing News")
                except Exception as e:
                    logger.warning(f"✗ Failed to fetch Bing News: {e}")
                    self.collection_stats['sources_failed'].append('bing_news')
        
        # ========================================================================
        # CRYPTO-SPECIFIC SOURCES (Always try CryptoCompare for crypto - it's free!)
        # ========================================================================
        is_crypto = '-USD' in self.symbol.upper() or any(crypto in self.symbol.upper() for crypto in ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOGE', 'DOT', 'MATIC'])
        if is_crypto:
            if 'all' in sources or 'cryptocompare' in sources:
                try:
                    logger.info("Attempting to fetch from CryptoCompare (free, no API key needed)...")
                    news = self._fetch_cryptocompare_news(days, max_articles=max_articles_per_source)
                    all_news.extend(news)
                    self.collection_stats['sources_used'].append('cryptocompare')
                    logger.info(f"✓ Collected {len(news)} articles from CryptoCompare")
                except Exception as e:
                    logger.warning(f"✗ Failed to fetch CryptoCompare news: {e}")
                    self.collection_stats['sources_failed'].append('cryptocompare')
        
        logger.info(f"Collection phase complete. Total articles collected so far: {len(all_news)}")
        
        # ========================================================================
        # PROCESSING: Remove duplicates and sort
        # ========================================================================
        self.collection_stats['total_collected'] = len(all_news)
        
        # Remove duplicates using multiple strategies
        seen = set()
        unique_news = []
        for item in all_news:
            # Create multiple keys for duplicate detection
            title_key = item.get('title', '').lower().strip()[:100]
            url_key = item.get('url', '')
            timestamp_key = item.get('timestamp', '')[:10]  # Date only
            
            # Check for duplicates
            key1 = (timestamp_key, title_key)
            key2 = url_key if url_key else None
            
            is_duplicate = False
            if key1 in seen:
                is_duplicate = True
            elif key2 and key2 in seen:
                is_duplicate = True
            
            if not is_duplicate:
                seen.add(key1)
                if key2:
                    seen.add(key2)
                unique_news.append(item)
            else:
                self.collection_stats['duplicates_removed'] += 1
        
        # Sort by timestamp
        unique_news.sort(key=lambda x: x.get('timestamp', ''))
        
        # Analyze sentiment for all news
        unique_news = self.analyze_sentiment(unique_news)
        
        self.news_cache = unique_news
        
        # Final summary
        logger.info("=" * 60)
        logger.info("NEWS COLLECTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Sources used: {self.collection_stats['sources_used']}")
        logger.info(f"Sources failed: {self.collection_stats['sources_failed']}")
        logger.info(f"Total collected (before dedup): {self.collection_stats['total_collected']}")
        logger.info(f"Duplicates removed: {self.collection_stats['duplicates_removed']}")
        logger.info(f"Final unique articles: {len(unique_news)}")
        logger.info("=" * 60)
        
        return unique_news
    
    # ============================================================================
    # ECONOMIC & FINANCIAL NEWS SOURCES
    # ============================================================================
    
    def _fetch_yfinance_news(self, days: int = 30, max_articles: int = 1000) -> List[Dict]:
        """Fetch news from yfinance (free, no API key needed)
        Note: yfinance typically returns 50-100 articles, this is a limitation of yfinance itself
        yfinance doesn't support date filtering, so we filter after fetching
        """
        news_list = []
        try:
            ticker = yf.Ticker(self.symbol)
            news = getattr(ticker, 'news', [])
            
            if not news:
                logger.warning("yfinance returned no news")
                return news_list
            
            logger.info(f"yfinance returned {len(news)} raw articles")
            
            # Calculate cutoff date for filtering
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # yfinance has its own limit (usually ~50-100), but we'll take what we can get
            for item in news[:max_articles * 2]:  # Get more to filter by date
                if len(news_list) >= max_articles:
                    break
                try:
                    provider_time = item.get('providerPublishTime', 0)
                    if not provider_time or provider_time == 0:
                        # If no timestamp, include it (yfinance limitation)
                        timestamp = datetime.now()
                    else:
                        timestamp = datetime.fromtimestamp(provider_time)
                    
                    # Filter by date if timestamp is valid
                    if provider_time > 0 and timestamp < cutoff_date:
                        continue
                    
                    news_list.append({
                        'timestamp': timestamp.isoformat(),
                        'title': item.get('title', ''),
                        'content': item.get('summary', ''),
                        'source': item.get('publisher', 'Yahoo Finance'),
                        'url': item.get('link', ''),
                        'category': 'financial',
                        'sentiment': 0.0
                    })
                except Exception as e:
                    continue
            
            logger.info(f"yfinance: Filtered to {len(news_list)} articles within {days} days (from {len(news)} raw articles)")
            
            if len(news_list) < 50:
                logger.warning(f"yfinance only returned {len(news_list)} articles - this is a yfinance limitation. Use other sources for more articles.")
        except Exception as e:
            logger.error(f"Error fetching yfinance news: {e}")
        
        return news_list
    
    def _fetch_newsapi_news(self, days: int, max_articles: int = 1000) -> List[Dict]:
        """Fetch news from NewsAPI with pagination support (requires API key)"""
        news_list = []
        api_key = os.getenv('NEWSAPI_KEY', '')
        
        if not api_key:
            return news_list
        
        try:
            query = self.symbol.replace('-USD', '').replace('-', ' ')
            url = "https://newsapi.org/v2/everything"
            
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)
            
            # NewsAPI supports pagination - fetch multiple pages
            page = 1
            page_size = 100  # Max per page
            total_pages = (max_articles + page_size - 1) // page_size
            
            while len(news_list) < max_articles and page <= total_pages:
                params = {
                    'q': f"{query} OR {self.symbol}",
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': min(page_size, max_articles - len(news_list)),
                    'page': page,
                    'from': from_date.strftime('%Y-%m-%d'),
                    'to': to_date.strftime('%Y-%m-%d'),
                    'apiKey': api_key
                }
                
                response = requests.get(url, params=params, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    articles = data.get('articles', [])
                    
                    if not articles:
                        break  # No more articles
                    
                    for article in articles:
                        if len(news_list) >= max_articles:
                            break
                        try:
                            timestamp = datetime.fromisoformat(
                                article['publishedAt'].replace('Z', '+00:00')
                            )
                            news_list.append({
                                'timestamp': timestamp.isoformat(),
                                'title': article.get('title', ''),
                                'content': article.get('description', '') or article.get('content', ''),
                                'source': article.get('source', {}).get('name', 'Unknown'),
                                'url': article.get('url', ''),
                                'category': 'economic',
                                'sentiment': 0.0
                            })
                        except Exception:
                            continue
                    
                    # Check if there are more pages
                    total_results = data.get('totalResults', 0)
                    if len(articles) < page_size or len(news_list) >= max_articles:
                        break
                    
                    page += 1
                elif response.status_code == 429:
                    logger.warning("NewsAPI rate limit exceeded")
                    break
                else:
                    logger.warning(f"NewsAPI returned status {response.status_code}")
                    break
        except Exception as e:
            logger.error(f"Error fetching NewsAPI news: {e}")
        
        return news_list
    
    def _fetch_alphavantage_news(self, days: int, max_articles: int = 1000) -> List[Dict]:
        """Fetch news from Alpha Vantage NEWS_SENTIMENT API"""
        news_list = []
        api_key = os.getenv('ALPHAVANTAGE_API_KEY', '')
        
        if not api_key:
            return news_list
        
        try:
            # Extract ticker from symbol
            ticker = self.symbol.split('-')[0] if '-' in self.symbol else self.symbol
            
            url = "https://www.alphavantage.co/query"
            # Alpha Vantage allows up to 1000, but we'll use max_articles parameter
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': ticker,
                'apikey': api_key,
                'limit': min(max_articles, 1000)  # Alpha Vantage max is 1000
            }
            
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                
                if 'feed' in data:
                    cutoff_date = datetime.now() - timedelta(days=days)
                    
                    for item in data['feed']:
                        if len(news_list) >= max_articles:
                            break
                        try:
                            # Parse timestamp
                            time_published = item.get('time_published', '')
                            if not time_published:
                                continue
                            
                            # Format: 20240101T123000
                            if len(time_published) >= 15:
                                year = int(time_published[0:4])
                                month = int(time_published[4:6])
                                day = int(time_published[6:8])
                                hour = int(time_published[9:11])
                                minute = int(time_published[11:13])
                                second = int(time_published[13:15])
                                timestamp = datetime(year, month, day, hour, minute, second)
                            else:
                                continue
                            
                            if timestamp < cutoff_date:
                                continue
                            
                            # Extract sentiment
                            sentiment_score = 0.0
                            if 'overall_sentiment_score' in item:
                                try:
                                    sentiment_score = float(item['overall_sentiment_score'])
                                except:
                                    pass
                            
                            news_list.append({
                                'timestamp': timestamp.isoformat(),
                                'title': item.get('title', ''),
                                'content': item.get('summary', '') or item.get('title', ''),
                                'source': item.get('source', 'Alpha Vantage'),
                                'url': item.get('url', ''),
                                'category': 'financial',
                                'sentiment': sentiment_score  # Already has sentiment
                            })
                        except Exception as e:
                            continue
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage news: {e}")
        
        return news_list
    
    def _fetch_finnhub_news(self, days: int, max_articles: int = 1000) -> List[Dict]:
        """Fetch news from Finnhub API"""
        news_list = []
        api_key = os.getenv('FINNHUB_API_KEY', '')
        
        if not api_key:
            return news_list
        
        try:
            is_crypto = '-USD' in self.symbol.upper()
            cutoff_timestamp = int((datetime.now() - timedelta(days=days)).timestamp())
            
            if is_crypto:
                # Crypto news
                url = "https://finnhub.io/api/v1/news"
                params = {'category': 'crypto', 'token': api_key}
            else:
                # Stock news
                ticker = self.symbol.split('-')[0]
                url = "https://finnhub.io/api/v1/company-news"
                params = {
                    'symbol': ticker,
                    'from': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
                    'to': datetime.now().strftime('%Y-%m-%d'),
                    'token': api_key
                }
            
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                
                if isinstance(data, list):
                    # Remove hard limit, use max_articles parameter
                    for item in data[:max_articles]:
                        if len(news_list) >= max_articles:
                            break
                        try:
                            ts = item.get('datetime', 0)
                            if ts and ts < cutoff_timestamp:
                                continue
                            
                            timestamp = datetime.fromtimestamp(ts) if ts else datetime.now()
                            
                            news_list.append({
                                'timestamp': timestamp.isoformat(),
                                'title': item.get('headline', ''),
                                'content': item.get('summary', '') or item.get('headline', ''),
                                'source': item.get('source', 'Finnhub'),
                                'url': item.get('url', ''),
                                'category': 'crypto' if is_crypto else 'financial',
                                'sentiment': 0.0
                            })
                        except Exception:
                            continue
        except Exception as e:
            logger.error(f"Error fetching Finnhub news: {e}")
        
        return news_list
    
    def _fetch_polygon_news(self, days: int) -> List[Dict]:
        """Fetch news from Polygon.io API"""
        news_list = []
        api_key = os.getenv('POLYGON_API_KEY', '')
        
        if not api_key:
            return news_list
        
        try:
            ticker = self.symbol.split('-')[0] if '-' in self.symbol else self.symbol
            url = f"https://api.polygon.io/v2/reference/news"
            
            params = {
                'ticker': ticker,
                'limit': 100,
                'apiKey': api_key
            }
            
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                
                cutoff_date = datetime.now() - timedelta(days=days)
                
                for item in data.get('results', []):
                    try:
                        published_utc = item.get('published_utc', '')
                        if not published_utc:
                            continue
                        
                        timestamp = datetime.fromisoformat(published_utc.replace('Z', '+00:00'))
                        if timestamp < cutoff_date:
                            continue
                        
                        news_list.append({
                            'timestamp': timestamp.isoformat(),
                            'title': item.get('title', ''),
                            'content': item.get('description', '') or item.get('title', ''),
                            'source': item.get('publisher', {}).get('name', 'Polygon'),
                            'url': item.get('article_url', ''),
                            'category': 'financial',
                            'sentiment': 0.0
                        })
                    except Exception:
                        continue
        except Exception as e:
            logger.error(f"Error fetching Polygon news: {e}")
        
        return news_list
    
    # ============================================================================
    # SOCIAL MEDIA SOURCES
    # ============================================================================
    
    def _fetch_twitter_news(self, days: int) -> List[Dict]:
        """Fetch news from Twitter/X (requires API keys)"""
        news_list = []
        
        # Check for Twitter API credentials
        bearer_token = os.getenv('TWITTER_BEARER_TOKEN', '')
        api_key = os.getenv('TWITTER_API_KEY', '')
        
        if not bearer_token and not api_key:
            return news_list
        
        try:
            query = self.symbol.replace('-USD', '').replace('-', ' ')
            
            # Use Twitter API v2 if bearer token available
            if bearer_token:
                url = "https://api.twitter.com/2/tweets/search/recent"
                headers = {'Authorization': f'Bearer {bearer_token}'}
                params = {
                    'query': f"{query} OR ${self.symbol} -is:retweet lang:en",
                    'max_results': 100,
                    'tweet.fields': 'created_at,public_metrics,author_id'
                }
                
                response = requests.get(url, headers=headers, params=params, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    cutoff_date = datetime.now() - timedelta(days=days)
                    
                    for tweet in data.get('data', []):
                        try:
                            created_at = datetime.fromisoformat(
                                tweet['created_at'].replace('Z', '+00:00')
                            )
                            if created_at < cutoff_date:
                                continue
                            
                            text = tweet.get('text', '')
                            news_list.append({
                                'timestamp': created_at.isoformat(),
                                'title': text[:100] + '...' if len(text) > 100 else text,
                                'content': text,
                                'source': 'Twitter',
                                'url': f"https://twitter.com/i/web/status/{tweet.get('id', '')}",
                                'category': 'social',
                                'sentiment': 0.0,
                                'engagement': tweet.get('public_metrics', {}).get('like_count', 0)
                            })
                        except Exception:
                            continue
        except Exception as e:
            logger.error(f"Error fetching Twitter news: {e}")
        
        return news_list
    
    def _fetch_reddit_news(self, days: int) -> List[Dict]:
        """Fetch news from Reddit (requires PRAW or API)"""
        news_list = []
        
        # Try using PRAW if available
        try:
            import praw
        except ImportError:
            return news_list
        
        try:
            reddit_client_id = os.getenv('REDDIT_CLIENT_ID', '')
            reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET', '')
            reddit_user_agent = os.getenv('REDDIT_USER_AGENT', 'StockNewsBot/1.0')
            
            if not reddit_client_id or not reddit_client_secret:
                return news_list
            
            reddit = praw.Reddit(
                client_id=reddit_client_id,
                client_secret=reddit_client_secret,
                user_agent=reddit_user_agent
            )
            
            query = self.symbol.replace('-USD', '').replace('-', ' ')
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Search relevant subreddits
            subreddits = ['wallstreetbets', 'stocks', 'investing', 'StockMarket']
            if '-USD' in self.symbol.upper():
                subreddits.extend(['cryptocurrency', 'CryptoCurrency', 'Bitcoin'])
            
            for subreddit_name in subreddits:
                try:
                    subreddit = reddit.subreddit(subreddit_name)
                    for submission in subreddit.search(query, limit=25, sort='new'):
                        try:
                            created_at = datetime.fromtimestamp(submission.created_utc)
                            if created_at < cutoff_date:
                                continue
                            
                            news_list.append({
                                'timestamp': created_at.isoformat(),
                                'title': submission.title[:200],
                                'content': submission.selftext[:500] if submission.selftext else submission.title,
                                'source': f'Reddit r/{subreddit_name}',
                                'url': f"https://reddit.com{submission.permalink}",
                                'category': 'social',
                                'sentiment': 0.0,
                                'engagement': submission.score
                            })
                        except Exception:
                            continue
                except Exception as e:
                    continue
        except Exception as e:
            logger.error(f"Error fetching Reddit news: {e}")
        
        return news_list
    
    # ============================================================================
    # LOCAL & REGIONAL NEWS
    # ============================================================================
    
    def _fetch_newscatcher_news(self, days: int) -> List[Dict]:
        """Fetch news from NewsCatcher API (local/regional news)"""
        news_list = []
        api_key = os.getenv('NEWSCATCHER_API_KEY', '')
        
        if not api_key:
            return news_list
        
        try:
            query = self.symbol.replace('-USD', '').replace('-', ' ')
            url = "https://api.newscatcher.ai/v1/search"
            
            headers = {'X-API-KEY': api_key}
            params = {
                'q': query,
                'lang': 'en',
                'page_size': 100,
                'from': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
                'to': datetime.now().strftime('%Y-%m-%d')
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                
                for article in data.get('articles', []):
                    try:
                        published_date = article.get('published_date', '')
                        if not published_date:
                            continue
                        
                        timestamp = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
                        
                        news_list.append({
                            'timestamp': timestamp.isoformat(),
                            'title': article.get('title', ''),
                            'content': article.get('summary', '') or article.get('title', ''),
                            'source': article.get('clean_url', 'Unknown'),
                            'url': article.get('link', ''),
                            'category': 'local',
                            'sentiment': 0.0
                        })
                    except Exception:
                        continue
        except Exception as e:
            logger.error(f"Error fetching NewsCatcher news: {e}")
        
        return news_list
    
    def _fetch_bing_news(self, days: int) -> List[Dict]:
        """Fetch news from Bing News Search API"""
        news_list = []
        api_key = os.getenv('BING_SEARCH_API_KEY', '')
        
        if not api_key:
            return news_list
        
        try:
            query = self.symbol.replace('-USD', '').replace('-', ' ')
            url = "https://api.bing.microsoft.com/v7.0/news/search"
            
            headers = {'Ocp-Apim-Subscription-Key': api_key}
            params = {
                'q': query,
                'count': 100,
                'freshness': 'Day' if days <= 1 else 'Week' if days <= 7 else 'Month'
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                cutoff_date = datetime.now() - timedelta(days=days)
                
                for article in data.get('value', []):
                    try:
                        date_published = article.get('datePublished', '')
                        if not date_published:
                            continue
                        
                        timestamp = datetime.fromisoformat(date_published.replace('Z', '+00:00'))
                        if timestamp < cutoff_date:
                            continue
                        
                        news_list.append({
                            'timestamp': timestamp.isoformat(),
                            'title': article.get('name', ''),
                            'content': article.get('description', ''),
                            'source': article.get('provider', [{}])[0].get('name', 'Bing News'),
                            'url': article.get('url', ''),
                            'category': 'local',
                            'sentiment': 0.0
                        })
                    except Exception:
                        continue
        except Exception as e:
            logger.error(f"Error fetching Bing News: {e}")
        
        return news_list
    
    # ============================================================================
    # CRYPTO-SPECIFIC SOURCES
    # ============================================================================
    
    def _fetch_cryptocompare_news(self, days: int, max_articles: int = 1000) -> List[Dict]:
        """Fetch news from CryptoCompare API (free, no API key required)"""
        news_list = []
        api_key = os.getenv('CRYPTOCOMPARE_API_KEY', '')
        
        # CryptoCompare has a free tier that works without API key
        try:
            symbol_clean = self.symbol.replace('-USD', '').upper()
            url = "https://min-api.cryptocompare.com/data/v2/news/"
            
            params = {
                'categories': symbol_clean if symbol_clean in ['BTC', 'ETH', 'LTC'] else 'ALL',
                'lang': 'EN'
            }
            
            if api_key:
                params['api_key'] = api_key
            
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                cutoff_date = datetime.now() - timedelta(days=days)
                
                # Remove hard limit, use max_articles parameter
                articles = data.get('Data', [])
                logger.info(f"CryptoCompare returned {len(articles)} articles, filtering by date and limit...")
                
                for article in articles[:max_articles * 2]:  # Get more to filter by date
                    if len(news_list) >= max_articles:
                        break
                    try:
                        published_on = article.get('published_on', 0)
                        if not published_on:
                            continue
                        
                        timestamp = datetime.fromtimestamp(published_on)
                        if timestamp < cutoff_date:
                            continue
                        
                        news_list.append({
                            'timestamp': timestamp.isoformat(),
                            'title': article.get('title', ''),
                            'content': article.get('body', '')[:500] if article.get('body') else article.get('title', ''),
                            'source': article.get('source', 'CryptoCompare'),
                            'url': article.get('url', ''),
                            'category': 'crypto',
                            'sentiment': 0.0
                        })
                    except Exception as e:
                        continue
                
                logger.info(f"CryptoCompare: Filtered to {len(news_list)} articles within {days} days")
            else:
                logger.warning(f"CryptoCompare returned status {response.status_code}")
        except Exception as e:
            logger.error(f"Error fetching CryptoCompare news: {e}")
        
        return news_list
    
    # ============================================================================
    # SENTIMENT ANALYSIS
    # ============================================================================
    
    def analyze_sentiment(self, news_list: List[Dict]) -> List[Dict]:
        """
        Analyze sentiment of news articles using enhanced keyword-based approach
        In production, this would use FinBERT or similar models
        """
        # Enhanced keyword lists
        positive_keywords = [
            'surge', 'rally', 'gain', 'bullish', 'up', 'rise', 'growth', 'increase',
            'soar', 'jump', 'climb', 'advance', 'profit', 'success', 'breakthrough',
            'milestone', 'record', 'high', 'peak', 'boom', 'expansion', 'positive'
        ]
        
        negative_keywords = [
            'crash', 'drop', 'fall', 'bearish', 'down', 'decline', 'loss', 'decrease',
            'plunge', 'sink', 'slump', 'collapse', 'crisis', 'failure', 'warning',
            'concern', 'risk', 'low', 'bottom', 'bust', 'recession', 'negative'
        ]
        
        for news in news_list:
            # Skip if sentiment already calculated (e.g., from Alpha Vantage)
            if news.get('sentiment', 0) != 0 and news.get('category') == 'financial':
                continue
            
            text = (news.get('title', '') + ' ' + news.get('content', '')).lower()
            
            # Count keyword matches
            pos_count = sum(1 for kw in positive_keywords if kw in text)
            neg_count = sum(1 for kw in negative_keywords if kw in text)
            
            # Calculate sentiment score
            if pos_count > neg_count:
                news['sentiment'] = min(1.0, pos_count / 10.0)
            elif neg_count > pos_count:
                news['sentiment'] = max(-1.0, -neg_count / 10.0)
            else:
                news['sentiment'] = 0.0
        
        return news_list


class TechnicalAnalysisAgent:
    """Agent for calculating technical indicators"""
    
    def __init__(self):
        self.indicators_cache = {}
    
    def calculate_indicators(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        df = price_data.copy()
        
        # Price-based indicators
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            if len(df) > period:
                df[f'sma_{period}'] = df['close'].rolling(period).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # Momentum indicators
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        df['macd'], df['macd_signal'], df['macd_hist'] = self._calculate_macd(df['close'])
        df['stoch_k'], df['stoch_d'] = self._calculate_stochastic(df, 14)
        
        # Volatility indicators
        df['atr'] = self._calculate_atr(df, 14)
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'], 20)
        df['volatility'] = df['returns'].rolling(20).std()
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-8)
        df['obv'] = self._calculate_obv(df)
        
        # Support/Resistance levels
        df['support'] = df['low'].rolling(20).min()
        df['resistance'] = df['high'].rolling(20).max()
        
        # Price position
        df['price_position'] = (df['close'] - df['support']) / (df['resistance'] - df['support'] + 1e-8)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    def _calculate_stochastic(self, df: pd.DataFrame, period: int = 14) -> Tuple:
        """Calculate Stochastic Oscillator"""
        low_min = df['low'].rolling(period).min()
        high_max = df['high'].rolling(period).max()
        stoch_k = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-8)
        stoch_d = stoch_k.rolling(3).mean()
        return stoch_k, stoch_d
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        return obv


class MacroEconomicAgent:
    """Agent for collecting macroeconomic data"""
    
    def __init__(self):
        self.macro_cache = {}
    
    def get_fear_greed_index(self) -> float:
        """Get Crypto Fear & Greed Index"""
        try:
            response = requests.get("https://api.alternative.me/fng/", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return float(data['data'][0]['value'])
        except Exception as e:
            logger.warning(f"Failed to fetch Fear & Greed Index: {e}")
        return 50.0  # Neutral
    
    def get_dxy(self) -> Optional[float]:
        """Get US Dollar Index"""
        try:
            ticker = yf.Ticker("DX-Y.NYB")
            data = ticker.history(period="1d", interval="1d")
            if not data.empty:
                return float(data['Close'].iloc[-1])
        except Exception as e:
            logger.warning(f"Failed to fetch DXY: {e}")
        return None


class MultiFactorAgent:
    """Agent for collecting broad multi-factor market context (proxy-based)."""

    def __init__(self):
        self.global_symbols = [
            '^GSPC', '^IXIC', '^DJI', '^RUT', '^VIX',
            'GC=F', 'SI=F', 'CL=F', 'HG=F',
            '^TNX', '^FVX', 'DX-Y.NYB', 'EURUSD=X', 'JPY=X',
            'TLT', 'HYG', 'LQD', 'XLK', 'XLF', 'XLE', 'XLY', 'XLP'
        ]
        self.crypto_symbols = ['BTC-USD', 'ETH-USD']

    def _safe_ticker_info(self, symbol: str) -> Dict[str, float]:
        fundamentals = {}
        try:
            info = yf.Ticker(symbol).info or {}
            fundamental_keys = [
                'marketCap', 'enterpriseValue', 'trailingPE', 'forwardPE', 'pegRatio',
                'priceToBook', 'priceToSalesTrailing12Months', 'debtToEquity',
                'returnOnEquity', 'returnOnAssets', 'profitMargins', 'operatingMargins',
                'freeCashflow', 'beta', 'sharesOutstanding', 'heldPercentInstitutions',
                'shortPercentOfFloat'
            ]
            for key in fundamental_keys:
                val = info.get(key)
                if val is not None:
                    fundamentals[f'fund_{key}'] = float(val)
        except Exception as e:
            logger.warning(f'Fundamental snapshot unavailable for {symbol}: {e}')
        return fundamentals

    def _safe_earnings_proxies(self, symbol: str) -> Dict[str, float]:
        proxies = {}
        try:
            ticker = yf.Ticker(symbol)
            cal = ticker.calendar
            if cal is not None and hasattr(cal, 'index') and len(cal.index) > 0:
                # yfinance calendar formats vary; keep robust and simple.
                idx_values = [str(i).lower() for i in cal.index]
                if any('earnings' in i for i in idx_values):
                    proxies['fund_has_earnings_calendar'] = 1.0
                else:
                    proxies['fund_has_earnings_calendar'] = 0.0
            else:
                proxies['fund_has_earnings_calendar'] = 0.0
        except Exception:
            proxies['fund_has_earnings_calendar'] = 0.0
        return proxies

    def collect_factor_context(self, symbol: str, period: str, interval: str) -> Dict[str, Any]:
        """Collect proxy factors for macro, cross-asset, credit, commodities, volatility, and fundamentals."""
        context = {
            'factor_panel': pd.DataFrame(),
            'fundamentals': {},
            'metadata': {'sources': []}
        }

        symbols = self.global_symbols.copy()
        if '-USD' not in symbol.upper():
            # For stocks add BTC/ETH as risk appetite proxies; for crypto we already have target coin.
            symbols.extend(self.crypto_symbols)

        try:
            panel = yf.download(
                tickers=symbols,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                group_by='ticker',
                threads=True
            )
            if not panel.empty:
                features = pd.DataFrame(index=panel.index)
                for sym in symbols:
                    try:
                        close = panel[(sym, 'Close')].astype(float)
                        vol = panel[(sym, 'Volume')].astype(float) if (sym, 'Volume') in panel.columns else None
                    except Exception:
                        continue
                    name = sym.replace('^', '').replace('=', '_').replace('-', '_').lower()
                    features[f'xf_{name}_close'] = close
                    features[f'xf_{name}_ret'] = close.pct_change()
                    if vol is not None:
                        features[f'xf_{name}_volchg'] = vol.pct_change()

                # Risk-on / risk-off and credit stress proxies.
                if 'xf_vix_close' in features.columns:
                    features['xf_vix_change'] = features['xf_vix_close'].pct_change()
                if 'xf_hyg_close' in features.columns and 'xf_lqd_close' in features.columns:
                    features['xf_credit_spread_proxy'] = (features['xf_hyg_close'] / (features['xf_lqd_close'] + 1e-8))
                if 'xf_tlt_close' in features.columns and 'xf_gspc_close' in features.columns:
                    features['xf_bond_equity_ratio'] = features['xf_tlt_close'] / (features['xf_gspc_close'] + 1e-8)

                features = features.replace([np.inf, -np.inf], np.nan).ffill().bfill()
                features.index = pd.to_datetime(features.index)
                features.reset_index(inplace=True)
                features.rename(columns={'index': 'datetime'}, inplace=True)
                context['factor_panel'] = features
                context['metadata']['sources'].append('yfinance_factor_panel')
        except Exception as e:
            logger.warning(f'Factor panel collection failed: {e}')

        # Fundamentals / structural factors (static values broadcast over time).
        fundamentals = self._safe_ticker_info(symbol)
        fundamentals.update(self._safe_earnings_proxies(symbol))
        context['fundamentals'] = fundamentals
        if fundamentals:
            context['metadata']['sources'].append('yfinance_fundamentals')

        return context


class RegimeDetectionAgent:
    """Agent for detecting market regimes"""
    
    def __init__(self):
        self.regime_history = deque(maxlen=100)
    
    def detect_regime(self, price_data: pd.DataFrame, lookback: int = 20) -> MarketRegime:
        """Detect current market regime"""
        if len(price_data) < lookback:
            return MarketRegime.UNKNOWN
        
        recent = price_data.tail(lookback)
        returns = recent['close'].pct_change().dropna()
        
        # Calculate metrics
        mean_return = returns.mean()
        volatility = returns.std()
        trend = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]
        
        # Regime classification
        if trend > 0.05 and volatility < 0.02:
            regime = MarketRegime.BULL
        elif trend < -0.05 and volatility < 0.02:
            regime = MarketRegime.BEAR
        elif volatility > 0.03:
            regime = MarketRegime.VOLATILE
        elif abs(trend) < 0.02:
            regime = MarketRegime.SIDEWAYS
        else:
            regime = MarketRegime.UNKNOWN
        
        self.regime_history.append(regime)
        return regime


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

class FeatureEngineer:
    """Comprehensive feature engineering pipeline"""
    
    def __init__(self):
        self.feature_scalers = {}
        self.feature_names = []
    
    def create_features(self, price_data: pd.DataFrame, news_data: List[Dict],
                       technical_data: pd.DataFrame, macro_data: Dict,
                       factor_context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Create comprehensive feature set"""
        
        # Start with technical indicators
        features = technical_data.copy()
        
        # Time-based features
        features = self._add_time_features(features)
        
        # Lag features
        features = self._add_lag_features(features, lags=[1, 2, 3, 5, 10, 20])
        
        # Rolling statistics
        features = self._add_rolling_features(features)
        
        # News sentiment features
        features = self._add_news_features(features, news_data)
        
        # Macro features
        features = self._add_macro_features(features, macro_data)
        
        # Cross-asset features and additional factor proxies
        features = self._add_cross_asset_features(features)
        features = self._add_factor_panel_features(features, factor_context)
        features = self._add_fundamental_features(features, factor_context)
        features = self._add_market_structure_proxies(features)
        
        # Remove NaN rows
        features = features.dropna()
        
        self.feature_names = [col for col in features.columns 
                             if col not in ['datetime', 'open', 'high', 'low', 'close', 'volume']]
        
        return features
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based cyclical features"""
        # Check for datetime column (handle different names)
        datetime_col = None
        for col in ['datetime', 'date', 'Date', 'Datetime']:
            if col in df.columns:
                datetime_col = col
                break
        
        if datetime_col is None:
            logger.warning("No datetime column found, skipping time features")
            return df
        
        # Ensure it's named 'datetime'
        if datetime_col != 'datetime':
            df.rename(columns={datetime_col: 'datetime'}, inplace=True)
        
        try:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df['hour'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.dayofweek
            df['day_of_month'] = df['datetime'].dt.day
            df['month'] = df['datetime'].dt.month
        except Exception as e:
            logger.warning(f"Error processing datetime: {e}, skipping time features")
            return df
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Binary features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_month_end'] = (df['day_of_month'] >= 25).astype(int)
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
        """Add lagged features"""
        for lag in lags:
            if 'close' in df.columns:
                df[f'close_lag_{lag}'] = df['close'].shift(lag)
                df[f'return_lag_{lag}'] = df['returns'].shift(lag) if 'returns' in df.columns else None
            if 'volume' in df.columns:
                df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window statistics"""
        windows = [5, 10, 20]
        
        for window in windows:
            if 'close' in df.columns:
                df[f'close_rolling_mean_{window}'] = df['close'].rolling(window).mean()
                df[f'close_rolling_std_{window}'] = df['close'].rolling(window).std()
                df[f'close_rolling_min_{window}'] = df['close'].rolling(window).min()
                df[f'close_rolling_max_{window}'] = df['close'].rolling(window).max()
            
            if 'returns' in df.columns:
                df[f'returns_rolling_mean_{window}'] = df['returns'].rolling(window).mean()
                df[f'returns_rolling_std_{window}'] = df['returns'].rolling(window).std()
        
        return df
    
    def _add_news_features(self, df: pd.DataFrame, news_data: List[Dict]) -> pd.DataFrame:
        """Add news sentiment features"""
        if not news_data:
            df['news_sentiment'] = 0.0
            df['news_count'] = 0.0
            return df
        
        # Check if datetime column exists
        datetime_col = None
        for col in ['datetime', 'date', 'Date', 'Datetime']:
            if col in df.columns:
                datetime_col = col
                break
        
        if datetime_col is None:
            # If no datetime column, add default news features
            df['news_sentiment'] = 0.0
            df['news_count'] = 0.0
            logger.warning("No datetime column found, using default news features")
            return df
        
        # Ensure datetime column is named 'datetime'
        if datetime_col != 'datetime':
            df.rename(columns={datetime_col: 'datetime'}, inplace=True)
        
        try:
            news_df = pd.DataFrame(news_data)
            if 'timestamp' not in news_df.columns or news_df.empty:
                df['news_sentiment'] = 0.0
                df['news_count'] = 0.0
                return df
            
            news_df['timestamp'] = pd.to_datetime(news_df['timestamp'])
            news_df['hour'] = news_df['timestamp'].dt.floor('H')
            
            df['datetime'] = pd.to_datetime(df['datetime'])
            df['hour'] = df['datetime'].dt.floor('H')
            
            # Aggregate news by hour
            hourly_news = news_df.groupby('hour').agg({
                'sentiment': ['mean', 'sum', 'count']
            }).reset_index()
            hourly_news.columns = ['hour', 'news_sentiment_mean', 'news_sentiment_sum', 'news_count']
            
            # Merge with main dataframe
            df = df.merge(hourly_news, on='hour', how='left')
            df['news_sentiment_mean'] = df['news_sentiment_mean'].fillna(0.0)
            df['news_sentiment_sum'] = df['news_sentiment_sum'].fillna(0.0)
            df['news_count'] = df['news_count'].fillna(0.0)
            
            # Rolling news sentiment
            df['news_sentiment_rolling'] = df['news_sentiment_mean'].rolling(24).mean()
        except Exception as e:
            logger.warning(f"Error adding news features: {e}, using defaults")
            df['news_sentiment'] = 0.0
            df['news_count'] = 0.0
        
        return df
    
    def _add_macro_features(self, df: pd.DataFrame, macro_data: Dict) -> pd.DataFrame:
        """Add macroeconomic features"""
        for key, value in macro_data.items():
            if value is not None:
                df[f'macro_{key}'] = value
        
        return df
    
    def _add_cross_asset_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cross-asset correlation features (placeholder)"""
        # In production, this would include correlations with other assets
        return df
    
    def _add_factor_panel_features(self, df: pd.DataFrame, factor_context: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """Merge external cross-asset factor panel (proxy for macro/flows/volatility/credit)."""
        if not factor_context:
            return df
        panel = factor_context.get('factor_panel')
        if panel is None or panel.empty:
            return df

        if 'datetime' not in df.columns:
            return df

        try:
            left = df.copy()
            left['datetime'] = pd.to_datetime(left['datetime'])
            right = panel.copy()
            right['datetime'] = pd.to_datetime(right['datetime'])

            left = left.sort_values('datetime')
            right = right.sort_values('datetime')

            merged = pd.merge_asof(
                left,
                right,
                on='datetime',
                direction='backward'
            )
            return merged
        except Exception as e:
            logger.warning(f'Failed to merge factor panel features: {e}')
            return df

    def _add_fundamental_features(self, df: pd.DataFrame, factor_context: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """Broadcast static fundamental proxies across rows."""
        if not factor_context:
            return df
        fundamentals = factor_context.get('fundamentals', {})
        for k, v in fundamentals.items():
            try:
                df[k] = float(v)
            except Exception:
                continue
        return df

    def _add_market_structure_proxies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add additional market structure proxies from OHLCV when direct feeds are unavailable."""
        if {'high', 'low', 'close'}.issubset(df.columns):
            df['range_pct'] = (df['high'] - df['low']) / (df['close'] + 1e-8)
            df['close_to_high'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)

        if 'returns' in df.columns:
            df['realized_vol_5'] = df['returns'].rolling(5).std()
            df['realized_vol_20'] = df['returns'].rolling(20).std()
            df['vol_term_structure'] = df['realized_vol_5'] / (df['realized_vol_20'] + 1e-8)
            df['downside_vol_20'] = df['returns'].where(df['returns'] < 0, 0).rolling(20).std()

        if 'volume' in df.columns:
            vma20 = df['volume'].rolling(20).mean()
            vma60 = df['volume'].rolling(60).mean()
            df['volume_pressure'] = vma20 / (vma60 + 1e-8)

        return df

    def normalize_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Normalize features"""
        feature_cols = [col for col in df.columns 
                       if col not in ['datetime', 'open', 'high', 'low', 'close', 'volume']]
        
        df_normalized = df.copy()
        
        for col in feature_cols:
            if col in df.columns:
                if fit:
                    mean = df[col].mean()
                    std = df[col].std() + 1e-8
                    self.feature_scalers[col] = {'mean': mean, 'std': std}
                else:
                    scaler = self.feature_scalers.get(col, {'mean': 0, 'std': 1})
                    mean = scaler['mean']
                    std = scaler['std']
                
                df_normalized[col] = (df[col] - mean) / std
        
        return df_normalized


# ============================================================================
# NEURAL NETWORK MODELS
# ============================================================================

class ShortTermModel(nn.Module):
    """Short-term prediction model (1-5 minutes) - Transformer based"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 4, num_heads: int = 8):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.pos_encoder = nn.Parameter(torch.randn(1, 1000, hidden_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = self.input_projection(x)
        x = x + self.pos_encoder[:, :seq_len, :]
        encoded = self.encoder(x)
        output = self.output_layer(encoded[:, -1, :])
        return output


class MediumTermModel(nn.Module):
    """Medium-term prediction model (1 hour - 1 day) - LSTM + Transformer hybrid"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        transformer_out = self.transformer(lstm_out)
        output = self.output_layer(transformer_out[:, -1, :])
        return output


class LongTermModel(nn.Module):
    """Long-term prediction model (1 day - 1 week) - Transformer with cross-attention"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 4):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.pos_encoder = nn.Parameter(torch.randn(1, 1000, hidden_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = self.input_projection(x)
        x = x + self.pos_encoder[:, :seq_len, :]
        encoded = self.encoder(x)
        output = self.output_layer(encoded[:, -1, :])
        return output


class RegimeClassifier(nn.Module):
    """Market regime classification model"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, len(MarketRegime))
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


# ============================================================================
# ENSEMBLE SYSTEM
# ============================================================================

class PredictionEnsemble:
    """Ensemble system combining multiple models"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.models = {}
        self.model_weights = {}
        self.regime_classifier = None
        
    def add_model(self, name: str, model: nn.Module, weight: float = 1.0):
        """Add a model to the ensemble"""
        self.models[name] = model.to(self.device)
        self.model_weights[name] = weight
    
    def set_regime_classifier(self, classifier: nn.Module):
        """Set regime classifier"""
        self.regime_classifier = classifier.to(self.device)
    
    def predict(self, features: torch.Tensor, regime: MarketRegime = None) -> Dict[str, torch.Tensor]:
        """Get predictions from all models"""
        predictions = {}
        
        with torch.no_grad():
            for name, model in self.models.items():
                model.eval()
                pred = model(features)
                predictions[name] = pred
        
        return predictions
    
    def ensemble_predict(self, features: torch.Tensor, regime: MarketRegime = None) -> torch.Tensor:
        """Combine predictions from all models"""
        predictions = self.predict(features, regime)
        
        # Adjust weights based on regime if classifier available
        weights = self._get_regime_weights(regime) if regime else self.model_weights
        
        # Weighted average
        weighted_sum = sum(predictions[name] * weights.get(name, 1.0) 
                          for name in predictions)
        total_weight = sum(weights.get(name, 1.0) for name in predictions)
        
        return weighted_sum / (total_weight + 1e-8)
    
    def _get_regime_weights(self, regime: MarketRegime) -> Dict[str, float]:
        """Get model weights based on regime"""
        # Default: equal weights, but can be customized per regime
        base_weights = self.model_weights.copy()
        
        if regime == MarketRegime.VOLATILE:
            # Favor short-term model in volatile markets
            base_weights['short_term'] = base_weights.get('short_term', 1.0) * 1.5
        elif regime == MarketRegime.BULL or regime == MarketRegime.BEAR:
            # Favor long-term model in trending markets
            base_weights['long_term'] = base_weights.get('long_term', 1.0) * 1.5
        
        return base_weights


# ============================================================================
# RISK MANAGEMENT
# ============================================================================

class RiskManager:
    """Risk management system"""
    
    def __init__(self, max_position_size: float = 0.1, max_drawdown: float = 0.2):
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown
        self.portfolio_value = 1.0
        self.peak_value = 1.0
    
    def validate_prediction(self, prediction: Prediction, current_price: float) -> bool:
        """Validate prediction for sanity"""
        # Check price bounds (e.g., ±50% change is suspicious)
        price_change = abs(prediction.price - current_price) / current_price
        if price_change > 0.5:
            logger.warning(f"Prediction change too large: {price_change:.2%}")
            return False
        
        # Check confidence
        if prediction.confidence < 0.3:
            logger.warning(f"Prediction confidence too low: {prediction.confidence:.2f}")
            return False
        
        return True
    
    def calculate_position_size(self, prediction: Prediction, current_price: float,
                               account_balance: float) -> float:
        """Calculate optimal position size using Kelly Criterion"""
        # Simplified Kelly Criterion
        win_prob = prediction.confidence
        win_loss_ratio = abs(prediction.price - current_price) / current_price
        
        if win_loss_ratio < 1e-8:
            return 0.0
        
        kelly_fraction = (win_prob * (1 + win_loss_ratio) - 1) / win_loss_ratio
        kelly_fraction = max(0, min(kelly_fraction, self.max_position_size))
        
        return kelly_fraction * account_balance
    
    def check_drawdown(self) -> bool:
        """Check if maximum drawdown is exceeded"""
        current_drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
        return current_drawdown < self.max_drawdown


# ============================================================================
# MAIN PREDICTION SYSTEM
# ============================================================================

class AIStockPredictor:
    """Main AI Stock Prediction System"""
    
    def __init__(self, symbol: str, device: Optional[torch.device] = None):
        self.symbol = symbol
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize agents
        self.market_agent = MarketDataAgent(symbol)
        self.news_agent = NewsSentimentAgent(symbol)
        self.technical_agent = TechnicalAnalysisAgent()
        self.macro_agent = MacroEconomicAgent()
        self.factor_agent = MultiFactorAgent()
        self.regime_agent = RegimeDetectionAgent()
        
        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer()
        
        # Initialize ensemble
        self.ensemble = PredictionEnsemble(self.device)
        
        # Initialize risk manager
        self.risk_manager = RiskManager()
        
        # Data storage
        self.price_data = None
        self.features = None
        self.models_trained = False
        self.sequence_length = 60
        self.using_fallback_data = False
        self.feature_cols = []
        self.normalization_stats = {
            'X_mean': None,
            'X_std': None,
            'y_mean': None,
            'y_std': None
        }
        self.model_validation_scores = {}
        
        logger.info(f"AI Stock Predictor initialized for {symbol} on {self.device}")
    
    def collect_data(self, period: str = '1y', interval: str = '1h', 
                    news_days: int = 30, max_news_per_source: int = 1000):
        """Collect all necessary data"""
        logger.info("Collecting data from all agents...")
        
        # Market data
        self.price_data = self.market_agent.collect_price_data(period, interval)
        self.using_fallback_data = (self.market_agent.last_data_source == 'synthetic')
        
        # News data
        logger.info(f"Collecting news: days={news_days}, max_per_source={max_news_per_source}")
        news_data = self.news_agent.collect_news(
            days=news_days, 
            max_articles_per_source=max_news_per_source,
            sources=['all'],
            include_social=True,
            include_local=True
        )
        # Note: analyze_sentiment is already called inside collect_news
        # But we'll call it again if needed (it's idempotent)
        if news_data:
            news_data = self.news_agent.analyze_sentiment(news_data)
        
        # Verify cache is set
        cache_length = len(self.news_agent.news_cache) if self.news_agent.news_cache else 0
        logger.info(f"News collection complete: {len(news_data)} articles returned, {cache_length} in cache")
        
        if cache_length != len(news_data):
            logger.warning(f"Cache mismatch! Returned {len(news_data)} but cache has {cache_length}")
            # Ensure cache matches returned data
            self.news_agent.news_cache = news_data
        
        # Technical indicators
        technical_data = self.technical_agent.calculate_indicators(self.price_data)
        
        # Macro data
        macro_data = {
            'fear_greed': self.macro_agent.get_fear_greed_index(),
            'dxy': self.macro_agent.get_dxy()
        }
        
        factor_context = self.factor_agent.collect_factor_context(
            symbol=self.symbol,
            period=period,
            interval=interval
        )

        # Feature engineering
        self.features = self.feature_engineer.create_features(
            self.price_data, news_data, technical_data, macro_data, factor_context
        )
        
        logger.info(f"Data collection complete. Features: {len(self.features.columns)}")
        return self.features
    
    def train_models(self, epochs: int = 50, batch_size: int = 32, lr: float = 0.001):
        """Train all models"""
        if self.features is None:
            raise ValueError("Must collect data first")
        
        logger.info("Training models...")
        
        # Prepare data
        self.feature_cols = [col for col in self.features.columns 
                             if col not in ['datetime', 'open', 'high', 'low', 'close', 'volume']]
        
        X = self.features[self.feature_cols].values
        y = self.features['close'].values
        
        # Create sequences
        X_seq, y_seq, prev_close_seq = self._create_sequences(X, y, self.sequence_length)
        if len(X_seq) < 30:
            raise ValueError("Insufficient sequence data for robust training")

        split_idx = max(1, int(len(X_seq) * 0.8))
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
        prev_train, prev_val = prev_close_seq[:split_idx], prev_close_seq[split_idx:]
        
        # Normalize using only training window statistics to avoid leakage
        X_mean = X_train.mean(axis=(0, 1))
        X_std = X_train.std(axis=(0, 1)) + 1e-8
        X_train = (X_train - X_mean) / X_std
        X_val = (X_val - X_mean) / X_std if len(X_val) > 0 else X_val
        
        y_mean = y_train.mean()
        y_std = y_train.std() + 1e-8
        y_train = (y_train - y_mean) / y_std
        y_val = (y_val - y_mean) / y_std if len(y_val) > 0 else y_val
        prev_train = (prev_train - y_mean) / y_std
        prev_val = (prev_val - y_mean) / y_std if len(prev_val) > 0 else prev_val

        self.normalization_stats = {
            'X_mean': X_mean,
            'X_std': X_std,
            'y_mean': y_mean,
            'y_std': y_std
        }
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        prev_train_tensor = torch.FloatTensor(prev_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device) if len(X_val) > 0 else None
        y_val_tensor = torch.FloatTensor(y_val).to(self.device) if len(y_val) > 0 else None
        prev_val_tensor = torch.FloatTensor(prev_val).to(self.device) if len(prev_val) > 0 else None
        
        # Initialize models
        input_dim = X_train.shape[2]
        
        short_term_model = ShortTermModel(input_dim).to(self.device)
        medium_term_model = MediumTermModel(input_dim).to(self.device)
        long_term_model = LongTermModel(input_dim).to(self.device)
        
        # Train each model
        models_to_train = [
            ('short_term', short_term_model),
            ('medium_term', medium_term_model),
            ('long_term', long_term_model)
        ]
        
        self.model_validation_scores = {}
        self.ensemble.models = {}
        self.ensemble.model_weights = {}

        for name, model in models_to_train:
            logger.info(f"Training {name} model...")
            self._train_model(
                model,
                X_train_tensor,
                y_train_tensor,
                prev_train_tensor,
                X_val_tensor,
                y_val_tensor,
                prev_val_tensor,
                epochs,
                batch_size,
                lr
            )
            val_metrics = self._evaluate_model_on_validation(
                model, X_val_tensor, y_val_tensor, prev_val_tensor, y_mean, y_std
            )
            self.model_validation_scores[name] = val_metrics
            self.ensemble.add_model(name, model, weight=1.0)

        self._set_ensemble_weights_from_validation()
        
        self.models_trained = True
        logger.info("All models trained successfully")
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray, seq_len: int) -> Tuple:
        """Create sequences for time series"""
        X_seq, y_seq, prev_close_seq = [], [], []
        for i in range(len(X) - seq_len):
            X_seq.append(X[i:i+seq_len])
            y_seq.append(y[i+seq_len])
            prev_close_seq.append(y[i+seq_len-1])
        return np.array(X_seq), np.array(y_seq), np.array(prev_close_seq)
    
    def _train_model(self, model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor,
                    prev_train: torch.Tensor,
                    X_val: Optional[torch.Tensor], y_val: Optional[torch.Tensor],
                    prev_val: Optional[torch.Tensor],
                    epochs: int, batch_size: int, lr: float):
        """Train a single model"""
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        direction_criterion = nn.BCEWithLogitsLoss()
        direction_weight = 0.15

        dataset = torch.utils.data.TensorDataset(X_train, y_train, prev_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        best_state = None
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y, batch_prev in dataloader:
                optimizer.zero_grad()
                pred = model(batch_X).squeeze()
                mse_loss = criterion(pred, batch_y)
                direction_target = (batch_y > batch_prev).float()
                direction_logits = pred - batch_prev
                direction_loss = direction_criterion(direction_logits, direction_target)
                loss = mse_loss + direction_weight * direction_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)

            if X_val is not None and y_val is not None and len(X_val) > 0:
                model.eval()
                with torch.no_grad():
                    val_pred = model(X_val).squeeze()
                    val_mse = criterion(val_pred, y_val)
                    if prev_val is not None:
                        val_direction_target = (y_val > prev_val).float()
                        val_direction_logits = val_pred - prev_val
                        val_direction = direction_criterion(val_direction_logits, val_direction_target)
                        val_loss = (val_mse + direction_weight * val_direction).item()
                    else:
                        val_loss = val_mse.item()
                model.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}, best val loss: {best_val_loss:.6f}")
                    break
            
            if (epoch + 1) % 10 == 0:
                if X_val is not None and y_val is not None and len(X_val) > 0:
                    logger.info(
                        f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}"
                    )
                else:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.6f}")

        if best_state is not None:
            model.load_state_dict(best_state)

    def _evaluate_model_on_validation(
        self,
        model: nn.Module,
        X_val: Optional[torch.Tensor],
        y_val: Optional[torch.Tensor],
        prev_val: Optional[torch.Tensor],
        y_mean: float,
        y_std: float
    ) -> Dict[str, float]:
        """Evaluate one trained model on validation slice for ensemble weighting."""
        if X_val is None or y_val is None or len(X_val) == 0:
            return {'rmse': 1.0, 'directional_accuracy': 0.5, 'score': 0.5}

        model.eval()
        with torch.no_grad():
            pred = model(X_val).squeeze()

        pred_prices = (pred.cpu().numpy() * y_std) + y_mean
        true_prices = (y_val.cpu().numpy() * y_std) + y_mean

        rmse = float(np.sqrt(np.mean((pred_prices - true_prices) ** 2)))

        if prev_val is not None:
            prev_prices = (prev_val.cpu().numpy() * y_std) + y_mean
            pred_direction = np.sign(pred_prices - prev_prices)
            true_direction = np.sign(true_prices - prev_prices)
            directional_accuracy = float(np.mean(pred_direction == true_direction))
        else:
            directional_accuracy = 0.5

        score = directional_accuracy / (1.0 + rmse)
        return {
            'rmse': rmse,
            'directional_accuracy': directional_accuracy,
            'score': float(score)
        }

    def _set_ensemble_weights_from_validation(self):
        """Set ensemble weights based on model validation quality."""
        if not self.model_validation_scores:
            return

        raw_weights = {}
        for name, metrics in self.model_validation_scores.items():
            raw_weights[name] = max(1e-4, metrics.get('score', 0.5))

        total = sum(raw_weights.values()) + 1e-8
        for name, value in raw_weights.items():
            self.ensemble.model_weights[name] = value / total
    

    def predict(self, timeframe: TimeFrame = TimeFrame.MEDIUM_TERM) -> Prediction:
        """Make a prediction"""
        if not self.models_trained:
            raise ValueError("Models must be trained first")
        
        if self.features is None or len(self.features) < self.sequence_length:
            raise ValueError("Insufficient data for prediction")

        if not self.feature_cols:
            self.feature_cols = [
                col for col in self.features.columns
                if col not in ['datetime', 'open', 'high', 'low', 'close', 'volume']
            ]

        if self.normalization_stats['X_mean'] is None:
            raise ValueError("Normalization statistics unavailable. Train models first.")

        # Get recent features
        recent_features = self.features[self.feature_cols].tail(self.sequence_length).values
        
        # Normalize with training statistics
        X_mean = self.normalization_stats['X_mean']
        X_std = self.normalization_stats['X_std']
        recent_features = (recent_features - X_mean) / X_std
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(recent_features).unsqueeze(0).to(self.device)
        
        # Detect regime
        regime = self.regime_agent.detect_regime(self.price_data)
        
        # Get predictions from all models
        predictions = self.ensemble.predict(X_tensor, regime)
        
        # Ensemble prediction
        ensemble_pred = self.ensemble.ensemble_predict(X_tensor, regime)
        
        # Denormalize
        current_price = self.price_data['close'].iloc[-1]
        y_mean = self.normalization_stats['y_mean']
        y_std = self.normalization_stats['y_std']
        predicted_price = ensemble_pred.item() * y_std + y_mean
        
        # Calculate confidence (based on model agreement)
        # Get normalized predictions from all models
        pred_values = [p.item() for p in predictions.values()]
        
        validation_directional = np.mean([
            metrics.get('directional_accuracy', 0.5)
            for metrics in self.model_validation_scores.values()
        ]) if self.model_validation_scores else 0.5

        if len(pred_values) < 2:
            # If only one model, use a default confidence
            confidence = 0.4 + 0.4 * validation_directional
        else:
            # Calculate coefficient of variation (CV) = std / mean
            # Lower CV = more agreement = higher confidence
            mean_abs = np.mean(np.abs(pred_values))
            std_abs = np.std(pred_values)
            
            if mean_abs < 1e-6:
                # If predictions are all very close to zero, models agree but magnitude is tiny
                # Check agreement by looking at relative spread
                if std_abs < 1e-6:
                    confidence = 0.7  # Models agree (all near zero)
                else:
                    confidence = 0.3  # Models disagree even at small scale
            else:
                # Coefficient of variation (normalized by mean)
                cv = std_abs / (mean_abs + 1e-8)
                
                agreement_confidence = 1.0 / (1.0 + cv)
                confidence = 0.6 * agreement_confidence + 0.4 * validation_directional
                
                # Additional boost if predictions are in similar direction
                signs = [1 if p > 0 else -1 for p in pred_values]
                if len(set(signs)) == 1:
                    # All models agree on direction
                    confidence = min(1.0, confidence * 1.1)
        
        # Clamp to [0, 1]
        confidence = max(0.0, min(1.0, confidence))
        
        # Calculate bounds (simplified)
        std_dev = np.std([p.item() * y_std for p in predictions.values()])
        upper_bound = predicted_price + 2 * std_dev
        lower_bound = predicted_price - 2 * std_dev
        
        # Feature importance (placeholder - would use SHAP in production)
        feature_importance = {name: 1.0 / len(self.feature_cols) for name in self.feature_cols[:10]}
        
        # Model contributions
        abs_sum = sum(abs(p.item()) for p in predictions.values()) + 1e-8
        model_contributions = {
            name: abs(pred.item()) / abs_sum
            for name, pred in predictions.items()
        }
        
        return Prediction(
            price=predicted_price,
            confidence=confidence,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            regime=regime,
            timeframe=timeframe,
            timestamp=datetime.now(),
            feature_importance=feature_importance,
            model_contributions=model_contributions
        )

    def evaluate_walk_forward(self, test_ratio: float = 0.2) -> Dict[str, float]:
        """Evaluate one-step-ahead performance using a holdout tail window."""
        if not self.models_trained:
            raise ValueError("Models must be trained first")
        if self.features is None:
            raise ValueError("Features are not available")

        if not self.feature_cols:
            self.feature_cols = [
                col for col in self.features.columns
                if col not in ['datetime', 'open', 'high', 'low', 'close', 'volume']
            ]

        X = self.features[self.feature_cols].values
        y = self.features['close'].values
        X_seq, y_seq, prev_close_seq = self._create_sequences(X, y, self.sequence_length)

        if len(X_seq) == 0:
            raise ValueError("Insufficient sequence data for evaluation")

        start_idx = max(1, int(len(X_seq) * (1 - test_ratio)))
        X_eval = X_seq[start_idx:]
        y_eval = y_seq[start_idx:]

        X_eval = (X_eval - self.normalization_stats['X_mean']) / self.normalization_stats['X_std']
        X_eval_tensor = torch.FloatTensor(X_eval).to(self.device)

        preds = []
        with torch.no_grad():
            for i in range(len(X_eval_tensor)):
                pred = self.ensemble.ensemble_predict(X_eval_tensor[i:i+1])
                preds.append(pred.item())

        y_mean = self.normalization_stats['y_mean']
        y_std = self.normalization_stats['y_std']
        pred_prices = np.array(preds) * y_std + y_mean
        true_prices = np.array(y_eval)

        mae = float(np.mean(np.abs(pred_prices - true_prices)))
        rmse = float(np.sqrt(np.mean((pred_prices - true_prices) ** 2)))
        mape = float(np.mean(np.abs((pred_prices - true_prices) / (true_prices + 1e-8))) * 100)

        prev_closes = y[start_idx + self.sequence_length - 1: start_idx + self.sequence_length - 1 + len(true_prices)]
        pred_direction = np.sign(pred_prices - prev_closes)
        true_direction = np.sign(true_prices - prev_closes)
        directional_accuracy = float(np.mean(pred_direction == true_direction))

        return {
            'samples': len(true_prices),
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'directional_accuracy': directional_accuracy
        }
    
    def save_system(self, path: str):
        """Save the entire system"""
        save_dict = {
            'models': {name: model.state_dict() for name, model in self.ensemble.models.items()},
            'feature_scalers': self.feature_engineer.feature_scalers,
            'feature_names': self.feature_engineer.feature_names,
            'symbol': self.symbol,
            'sequence_length': self.sequence_length,
            'feature_cols': self.feature_cols,
            'normalization_stats': self.normalization_stats
        }
        torch.save(save_dict, path)
        logger.info(f"System saved to {path}")
    
    def load_system(self, path: str):
        """Load a saved system"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Reconstruct models
        input_dim = len(checkpoint['feature_names'])
        
        short_term = ShortTermModel(input_dim).to(self.device)
        medium_term = MediumTermModel(input_dim).to(self.device)
        long_term = LongTermModel(input_dim).to(self.device)
        
        short_term.load_state_dict(checkpoint['models']['short_term'])
        medium_term.load_state_dict(checkpoint['models']['medium_term'])
        long_term.load_state_dict(checkpoint['models']['long_term'])
        
        self.ensemble.add_model('short_term', short_term, 1.0)
        self.ensemble.add_model('medium_term', medium_term, 1.0)
        self.ensemble.add_model('long_term', long_term, 1.0)
        
        self.feature_engineer.feature_scalers = checkpoint['feature_scalers']
        self.feature_engineer.feature_names = checkpoint['feature_names']
        self.sequence_length = checkpoint.get('sequence_length', 60)
        self.feature_cols = checkpoint.get('feature_cols', checkpoint['feature_names'])
        self.normalization_stats = checkpoint.get('normalization_stats', self.normalization_stats)
        self.models_trained = True
        
        logger.info(f"System loaded from {path}")
