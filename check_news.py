"""
뉴스 수집 동작 확인 스크립트
실행: python check_news.py
"""
import os
import sys

# .env 로드 (스크립트/프로젝트 폴더 기준)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_env_path = os.path.join(_script_dir, '.env')

def _load_env_fallback():
    """load_dotenv가 안 먹힐 때 .env를 직접 파싱 (BOM/인코딩/\\r 대비)"""
    want = ('NEWSAPI_KEY', 'ALPHAVANTAGE_API_KEY', 'FINNHUB_API_KEY')
    if all(os.getenv(k) for k in want):
        return
    if not os.path.isfile(_env_path):
        return
    _bom = chr(0xFEFF)
    raw = None
    for enc in ('utf-8-sig', 'utf-8', 'cp949', 'latin-1'):
        try:
            with open(_env_path, 'r', encoding=enc) as f:
                raw = f.read()
            break
        except Exception:
            continue
    if raw is None:
        return
    for line in raw.replace('\r\n', '\n').replace('\r', '\n').split('\n'):
        line = line.strip().replace(_bom, '')
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        k = k.strip().replace(_bom, '').replace('\r', '').strip()
        v = v.strip().strip('"').strip("'").replace('\r', '').strip()
        if k in want and not os.getenv(k) and v:
            os.environ[k] = v

try:
    from dotenv import load_dotenv
    load_dotenv(_env_path)
    load_dotenv()
except ImportError:
    pass
# dotenv가 .env를 못 읽었을 수 있으므로 한 번 더 시도
_load_env_fallback()

def _mask(s):
    if not s or len(s) < 5:
        return "(없음)" if not s else "****"
    return s[:2] + "*" * (len(s) - 4) + s[-2:]

def main():
    print("=" * 60)
    print("뉴스 수집 체크 (check_news.py)")
    print("=" * 60)
    
    # .env / API 키 로드 여부 확인
    print("\n[0] .env / API 키 확인")
    print(f"    .env 경로: {_env_path}")
    print(f"    .env 존재: {os.path.isfile(_env_path)}")
    
    # .env 파일에 실제로 적힌 키 이름 확인 (이름만, 값은 안 보여줌)
    if os.path.isfile(_env_path):
        try:
            with open(_env_path, 'r', encoding='utf-8-sig') as f:  # utf-8-sig: BOM 제거
                raw = f.read()
        except Exception as e:
            raw = f"(파일 읽기 실패: {e})"
        lines = [s.strip() for s in raw.splitlines() if s.strip() and not s.strip().startswith('#')]
        keys_in_file = []
        for line in lines:
            if '=' in line:
                key = line.split('=', 1)[0].strip()
                if key:
                    keys_in_file.append(repr(key))  # BOM 등 보이게
        if keys_in_file:
            print(f"    .env 안의 키 이름: {', '.join(keys_in_file)}")
        else:
            print("    .env 안의 키: (KEY=값 형식 줄이 없음)")
    
    for k in ('NEWSAPI_KEY', 'ALPHAVANTAGE_API_KEY', 'FINNHUB_API_KEY'):
        v = os.getenv(k, '')
        if v:
            print(f"    {k}: 로드됨 ({_mask(v)})")
        else:
            print(f"    {k}: (없음)")
    
    try:
        from go_stock import StockPriceGenerator
    except Exception as e:
        print(f"오류: go_stock 로드 실패 - {e}")
        sys.exit(1)
    
    gen = StockPriceGenerator(embedding_model='bert')  # bert는 가볍게
    symbol = "BTC-USD"
    
    # 1) yfinance (기본, API 키 불필요)
    print("\n[1] yfinance (기본)")
    try:
        news = gen.collect_news_data(symbol, days=30, news_source='yfinance')
        print(f"    수집: {len(news)}개")
        if news:
            print(f"    예시: {news[0].get('title', '')[:50]}...")
        else:
            print("    → 0건인 경우: Yahoo 뉴스 일시 불가/차단, 또는 해당 종목 뉴스 없음.")
    except Exception as e:
        print(f"    오류: {e}")
    
    # 2) NewsAPI
    print("\n[2] NewsAPI")
    if os.getenv('NEWSAPI_KEY'):
        try:
            news = gen.collect_news_data(symbol, days=30, news_source='newsapi')
            print(f"    수집: {len(news)}개")
            if news:
                print(f"    예시: {news[0].get('title', '')[:50]}...")
        except Exception as e:
            print(f"    오류: {e}")
    else:
        print("    NEWSAPI_KEY 미설정 → .env 또는 환경 변수 확인")
    
    # 3) Alpha Vantage
    print("\n[3] Alpha Vantage")
    if os.getenv('ALPHAVANTAGE_API_KEY'):
        try:
            news = gen.collect_news_data(symbol, days=30, news_source='alphavantage')
            print(f"    수집: {len(news)}개")
            if news:
                print(f"    예시: {news[0].get('title', '')[:50]}... (감정: {news[0].get('sentiment')})")
        except Exception as e:
            print(f"    오류: {e}")
    else:
        print("    ALPHAVANTAGE_API_KEY 미설정 → .env 또는 환경 변수 확인")
    
    # 4) Finnhub
    print("\n[4] Finnhub")
    if os.getenv('FINNHUB_API_KEY'):
        try:
            news = gen.collect_news_data(symbol, days=30, news_source='finnhub')
            print(f"    수집: {len(news)}개 (암호화폐는 category=crypto 사용)")
            if news:
                print(f"    예시: {news[0].get('title', '')[:50]}...")
        except Exception as e:
            print(f"    오류: {e}")
    else:
        print("    FINNHUB_API_KEY 미설정 → .env 또는 환경 변수 확인")
    
    print("\n" + "=" * 60)
    print("체크 완료. 0건인 이유는 NEWS_API_SETUP.md '문제 해결' 참고.")
    print("=" * 60)

if __name__ == "__main__":
    main()
