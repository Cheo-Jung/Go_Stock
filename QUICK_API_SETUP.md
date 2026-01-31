# Quick API Keys Setup Guide

## 📝 Step 1: Create `.env` File

1. Go to your project folder: `C:\Users\pstcw\Downloads\Go_Stock`
2. Create a new file named `.env` (just `.env` with nothing else)
3. Copy and paste the template below into the file

## 📋 Step 2: Copy This Template

Open your `.env` file and paste this:

```env
# Essential Free APIs (Get these first!)
NEWSAPI_KEY=your_key_here
ALPHAVANTAGE_API_KEY=your_key_here
FINNHUB_API_KEY=your_key_here

# Optional APIs
POLYGON_API_KEY=your_key_here
TWITTER_BEARER_TOKEN=your_key_here
REDDIT_CLIENT_ID=your_key_here
REDDIT_CLIENT_SECRET=your_key_here
REDDIT_USER_AGENT=StockNewsBot/1.0
NEWSCATCHER_API_KEY=your_key_here
BING_SEARCH_API_KEY=your_key_here
CRYPTOCOMPARE_API_KEY=your_key_here
```

## 🔗 Step 3: Get Your API Keys

### ⭐ Start with these 3 FREE APIs:

#### 1. NewsAPI (100 articles/day free)
- **Link**: https://newsapi.org/register
- **Steps**:
  1. Click "Get API Key"
  2. Sign up with email
  3. Verify email
  4. Copy your API key
  5. Paste in `.env`: `NEWSAPI_KEY=paste_key_here`

#### 2. Alpha Vantage (500 calls/day free)
- **Link**: https://www.alphavantage.co/support/#api-key
- **Steps**:
  1. Fill out the form
  2. Submit
  3. Check email for API key
  4. Paste in `.env`: `ALPHAVANTAGE_API_KEY=paste_key_here`

#### 3. Finnhub (60 calls/min free)
- **Link**: https://finnhub.io/register
- **Steps**:
  1. Sign up with email
  2. Verify email
  3. Copy API key from dashboard
  4. Paste in `.env`: `FINNHUB_API_KEY=paste_key_here`

## ✅ Step 4: Save and Test

1. Save the `.env` file
2. Restart your Streamlit app
3. Try collecting data again
4. Check the console logs to see which sources are working

## 📊 Expected Results

With just these 3 free APIs:
- **NewsAPI**: Up to 1000+ articles (with pagination)
- **Alpha Vantage**: Up to 1000 articles
- **Finnhub**: Up to 1000+ articles
- **yfinance**: ~50-100 articles (free, no key needed)
- **Total**: 3000-4000+ articles! 🎉

## 🔍 Quick Links Summary

| API | Link | Free Tier |
|-----|------|-----------|
| NewsAPI | https://newsapi.org/register | 100/day |
| Alpha Vantage | https://www.alphavantage.co/support/#api-key | 500/day |
| Finnhub | https://finnhub.io/register | 60/min |
| Reddit | https://www.reddit.com/prefs/apps | Unlimited |
| Twitter | https://developer.twitter.com/ | Limited |

## 💡 Tips

- Start with NewsAPI, Alpha Vantage, and Finnhub (all free!)
- CryptoCompare works without API key for crypto symbols
- Reddit is free but requires app setup
- Twitter requires developer account approval (takes 1-2 days)

## 🚨 Important

- **NEVER** share your `.env` file
- **NEVER** commit `.env` to Git (it's already in `.gitignore`)
- Keep your API keys secret!
