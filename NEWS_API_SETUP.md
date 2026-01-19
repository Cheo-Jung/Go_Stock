# 뉴스 API 설정 가이드

이 가이드는 Go_Stock 프로젝트에서 뉴스 데이터를 수집하는 방법을 설명합니다.

## 🎯 지원되는 뉴스 소스

프로젝트는 다음 4가지 뉴스 소스를 지원합니다:

| 소스 | 무료 여부 | API 키 필요 | 제한사항 | 권장도 |
|------|----------|------------|---------|--------|
| **yfinance** | ✅ 무료 | ❌ 불필요 | Yahoo Finance 스크래핑, 안정성 낮음 | ⭐⭐⭐ (기본값) |
| **NewsAPI** | ✅ 무료 티어 | ✅ 필요 | 100 requests/day | ⭐⭐⭐⭐ |
| **Alpha Vantage** | ✅ 무료 티어 | ✅ 필요 | 25 requests/day, 감정 분석 포함 | ⭐⭐⭐⭐⭐ |
| **Finnhub** | ✅ 무료 티어 | ✅ 필요 | 제한적 | ⭐⭐⭐⭐ |

---

## 🚀 빠른 시작 (yfinance - 기본값)

**가장 간단한 방법**: API 키 없이 바로 사용 가능

```python
from go_stock import StockPriceGenerator

generator = StockPriceGenerator()

# yfinance 사용 (기본값, API 키 불필요)
news_data = generator.collect_news_data('BTC-USD', days=30, news_source='yfinance')
print(f"수집된 뉴스: {len(news_data)}개")
```

---

## 📝 각 뉴스 소스 상세 설정

### 1. yfinance (기본값) ✅ **가장 간단**

**장점:**
- ✅ API 키 불필요
- ✅ 즉시 사용 가능
- ✅ 무료

**단점:**
- ⚠️ 안정성이 낮을 수 있음 (스크래핑 기반)
- ⚠️ 감정 분석 제공 안 함

**사용법:**
```python
news_data = generator.collect_news_data('BTC-USD', days=30, news_source='yfinance')
```

---

### 2. NewsAPI ⭐ **추천**

**무료 티어:**
- 100 requests/day
- 최대 1개월 이전 데이터

**API 키 발급:**
1. [NewsAPI.org](https://newsapi.org/register) 접속
2. 무료 계정 생성
3. API 키 복사

**설정:**
```bash
# Windows (PowerShell)
$env:NEWSAPI_KEY="your_api_key_here"

# Windows (CMD)
set NEWSAPI_KEY=your_api_key_here

# Linux/Mac
export NEWSAPI_KEY="your_api_key_here"

# Python 코드에서
import os
os.environ['NEWSAPI_KEY'] = 'your_api_key_here'
```

**사용법:**
```python
news_data = generator.collect_news_data('BTC-USD', days=30, news_source='newsapi')
```

---

### 3. Alpha Vantage ⭐⭐⭐ **최고 추천**

**무료 티어:**
- 25 requests/day
- **감정 분석 포함** (sentiment score)
- 실시간 뉴스

**API 키 발급:**
1. [Alpha Vantage](https://www.alphavantage.co/support/#api-key) 접속
2. 무료 API 키 발급 (이메일 입력만 필요)

**설정:**
```bash
# Windows (PowerShell)
$env:ALPHAVANTAGE_API_KEY="your_api_key_here"

# Windows (CMD)
set ALPHAVANTAGE_API_KEY=your_api_key_here

# Linux/Mac
export ALPHAVANTAGE_API_KEY="your_api_key_here"

# Python 코드에서
import os
os.environ['ALPHAVANTAGE_API_KEY'] = 'your_api_key_here'
```

**사용법:**
```python
news_data = generator.collect_news_data('BTC-USD', days=30, news_source='alphavantage')
```

**특징:**
- ✅ **감정 점수 포함** (-1.0 ~ 1.0)
- ✅ 금융 뉴스에 특화
- ✅ 실시간 데이터

---

### 4. Finnhub

**무료 티어:**
- 제한적 요청 수
- 금융 뉴스 특화

**API 키 발급:**
1. [Finnhub.io](https://finnhub.io/register) 접속
2. 무료 계정 생성
3. API 키 복사

**설정:**
```bash
# Windows (PowerShell)
$env:FINNHUB_API_KEY="your_api_key_here"

# Windows (CMD)
set FINNHUB_API_KEY=your_api_key_here

# Linux/Mac
export FINNHUB_API_KEY="your_api_key_here"
```

**사용법:**
```python
news_data = generator.collect_news_data('BTC-USD', days=30, news_source='finnhub')
```

---

## 💻 사용 예제

### 기본 사용 (yfinance)

```python
from go_stock import StockPriceGenerator

generator = StockPriceGenerator()

# 뉴스 수집
news_data = generator.collect_news_data('BTC-USD', days=30)

# 결과 확인
for news in news_data[:5]:
    print(f"제목: {news['title']}")
    print(f"시간: {news['timestamp']}")
    print(f"감정: {news['sentiment']}")
    print("---")
```

### Alpha Vantage 사용 (감정 분석 포함)

```python
import os
from go_stock import StockPriceGenerator

# API 키 설정
os.environ['ALPHAVANTAGE_API_KEY'] = 'your_api_key_here'

generator = StockPriceGenerator()

# 뉴스 수집 (감정 점수 포함)
news_data = generator.collect_news_data(
    'BTC-USD', 
    days=30, 
    news_source='alphavantage'
)

# 감정 점수 확인
for news in news_data:
    sentiment = news['sentiment']
    if sentiment > 0.3:
        print(f"긍정적: {news['title']} (점수: {sentiment:.2f})")
    elif sentiment < -0.3:
        print(f"부정적: {news['title']} (점수: {sentiment:.2f})")
```

### 여러 소스 자동 fallback

```python
from go_stock import StockPriceGenerator

generator = StockPriceGenerator()

# 여러 소스 시도
sources = ['alphavantage', 'newsapi', 'finnhub', 'yfinance']
news_data = []

for source in sources:
    try:
        news_data = generator.collect_news_data(
            'BTC-USD', 
            days=30, 
            news_source=source
        )
        if len(news_data) > 0:
            print(f"✓ {source}에서 {len(news_data)}개 뉴스 수집 성공!")
            break
    except Exception as e:
        print(f"⚠ {source} 실패: {e}")
        continue
```

---

## 🔧 환경 변수 설정 (영구적)

### Windows

**PowerShell (현재 세션):**
```powershell
$env:ALPHAVANTAGE_API_KEY="your_api_key_here"
```

**시스템 환경 변수 (영구적):**
1. `시스템 속성` → `고급` → `환경 변수`
2. `새로 만들기` → 변수 이름과 값 입력
3. 재시작 후 적용

### Linux/Mac

**현재 세션:**
```bash
export ALPHAVANTAGE_API_KEY="your_api_key_here"
```

**영구적 설정 (.bashrc 또는 .zshrc):**
```bash
echo 'export ALPHAVANTAGE_API_KEY="your_api_key_here"' >> ~/.bashrc
source ~/.bashrc
```

### Python 코드에서

```python
import os

# 코드 시작 부분에 추가
os.environ['ALPHAVANTAGE_API_KEY'] = 'your_api_key_here'
os.environ['NEWSAPI_KEY'] = 'your_api_key_here'
os.environ['FINNHUB_API_KEY'] = 'your_api_key_here'
```

### .env 파일 사용 (권장) ⭐

**⚠️ 코드에 API 키를 직접 적지 마세요!** `.env` 파일을 사용하세요.

1. 프로젝트 폴더에 `.env` 파일 생성
2. 내용 예: `NEWSAPI_KEY=your_key_here` (각 API별로 한 줄씩)
3. `pip install python-dotenv` 후 `go_stock` 실행 시 자동 로드
4. `.env`는 `.gitignore`에 포함되어 Git에 올라가지 않습니다.

**Google Colab**에서: 로컬과 같이 `.env`를 업로드하거나, Colab 셀에서 `.env`를 만들면 됩니다.  
→ `GOOGLE_COLAB_SETUP.md` **「4. API 키 설정」** 참고.

---

## 📊 뉴스 데이터 구조

수집된 뉴스 데이터는 다음 형식을 따릅니다:

```python
{
    'timestamp': '2024-01-15T10:30:00',  # ISO 형식
    'title': 'Bitcoin price surges...',   # 뉴스 제목
    'content': 'Bitcoin price...',        # 뉴스 내용/요약
    'sentiment': 0.65,                    # 감정 점수 (-1.0 ~ 1.0)
    'source': 'Reuters',                  # 출처
    'url': 'https://...'                  # 원문 URL
}
```

**참고:**
- `sentiment`: Alpha Vantage만 제공, 나머지는 0
- `content`: 요약이 없으면 제목 사용

---

## 🎯 권장 설정

### 시나리오별 추천

1. **빠른 테스트/프로토타입**
   ```python
   news_source='yfinance'  # API 키 불필요
   ```

2. **감정 분석이 필요한 경우**
   ```python
   news_source='alphavantage'  # 감정 점수 제공
   ```

3. **안정적인 프로덕션**
   ```python
   # 여러 소스 자동 fallback 구현
   ```

4. **대량 데이터 수집**
   ```python
   # 유료 API 고려 (NewsAPI Pro, Alpha Vantage Premium)
   ```

---

## ⚠️ 주의사항

1. **API 키 보안**
   - API 키를 코드에 하드코딩하지 마세요
   - 환경 변수 사용 권장
   - Git에 커밋하지 마세요

2. **Rate Limits**
   - 무료 티어는 요청 수 제한이 있습니다
   - 여러 종목을 수집할 때 주의하세요

3. **데이터 품질**
   - yfinance는 스크래핑 기반으로 안정성이 낮을 수 있습니다
   - 프로덕션에서는 유료 API 고려

---

## 🔍 문제 해결

### "API 키가 설정되지 않았습니다" 오류

```python
# 환경 변수 확인
import os
print(os.getenv('ALPHAVANTAGE_API_KEY'))  # None이면 설정 안 됨

# 코드에서 직접 설정
os.environ['ALPHAVANTAGE_API_KEY'] = 'your_key'
```

### 뉴스가 수집되지 않음

```python
# 1. 다른 소스 시도
news_data = generator.collect_news_data('BTC-USD', news_source='yfinance')

# 2. days 파라미터 조정 (더 짧은 기간)
news_data = generator.collect_news_data('BTC-USD', days=7)

# 3. 심볼 확인 (올바른 형식인지)
# 주식: 'AAPL', 'TSLA'
# 암호화폐: 'BTC-USD', 'ETH-USD'
```

### Rate Limit 오류

```python
# 요청 간격 추가
import time

for symbol in ['BTC-USD', 'ETH-USD']:
    news_data = generator.collect_news_data(symbol, days=30)
    time.sleep(1)  # 1초 대기
```

---

## 📚 참고 자료

- [NewsAPI 문서](https://newsapi.org/docs)
- [Alpha Vantage 문서](https://www.alphavantage.co/documentation/)
- [Finnhub 문서](https://finnhub.io/docs/api)
- [yfinance GitHub](https://github.com/ranaroussi/yfinance)

---

## 🎉 결론

**빠른 시작:**
- yfinance 사용 (API 키 불필요)

**최고 품질:**
- Alpha Vantage 사용 (감정 분석 포함)

**안정성:**
- 여러 소스 자동 fallback 구현

**지금 바로 시작하세요!** 🚀
