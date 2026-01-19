"""
주식/코인 가격 및 뉴스 데이터를 연동하여 새로운 가격을 생성하는 프로그램
LLM/Transformer 기반 시계열 생성 모델
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from datetime import datetime, timedelta
import yfinance as yf
import requests
from typing import List, Dict, Tuple, Optional
import json
import os
import warnings
warnings.filterwarnings('ignore')

# .env 파일에서 API 키 로드 (python-dotenv 설치 시)
# 스크립트 위치 기준으로 .env 찾기 (실행 경로에 상관없이 동작)
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')

def _load_env_fallback():
    """load_dotenv가 안 될 때 .env 직접 파싱 (BOM/인코딩/\\r 대비). Colab: /content, cwd도 확인."""
    want = ('NEWSAPI_KEY', 'ALPHAVANTAGE_API_KEY', 'FINNHUB_API_KEY')
    if all(os.getenv(k) for k in want):
        return
    _bom = chr(0xFEFF)
    paths = [
        _env_path,
        '/content/.env',  # Google Colab 기본
        os.path.join(os.getcwd(), '.env'),
    ]
    for p in paths:
        if not p or not os.path.isfile(p):
            continue
        raw = None
        for enc in ('utf-8-sig', 'utf-8', 'cp949', 'latin-1'):
            try:
                with open(p, 'r', encoding=enc) as f:
                    raw = f.read()
                break
            except Exception:
                continue
        if raw is None:
            continue
        for line in raw.replace('\r\n', '\n').replace('\r', '\n').split('\n'):
            line = line.strip().replace(_bom, '')
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            k = k.strip().replace(_bom, '').replace('\r', '').strip()
            v = v.strip().strip('"').strip("'").replace('\r', '').strip()
            if k in want and not os.getenv(k) and v:
                os.environ[k] = v
        if all(os.getenv(k) for k in want):
            break

try:
    from dotenv import load_dotenv
    load_dotenv(_env_path)
    load_dotenv()
    # Colab: 업로드한 .env가 /content 또는 cwd에 있는 경우
    for p in ('/content/.env', os.path.join(os.getcwd(), '.env')):
        if p and p != _env_path and os.path.isfile(p):
            load_dotenv(p)
            break
except ImportError:
    pass
_load_env_fallback()


class PriceNewsDataset(Dataset):
    """가격과 뉴스 데이터를 결합한 데이터셋"""
    
    # 지원되는 모델과 임베딩 차원
    MODEL_CONFIGS = {
        'fine5': {
            'name': 'intfloat/e5-mistral-7b-instruct',
            'embedding_dim': 4096,
            'description': 'Fin-E5 - 최고 성능 금융 임베딩 모델 (16GB+ GPU 필요, Colab 권장)'
        },
        'finbert': {
            'name': 'ProsusAI/finbert',
            'embedding_dim': 768,
            'description': 'FinBERT - 금융 텍스트에 최적화된 모델 (권장)'
        },
        'distilbert': {
            'name': 'distilbert-base-uncased',
            'embedding_dim': 768,
            'description': 'DistilBERT - 더 빠르고 작은 모델'
        },
        'bert': {
            'name': 'bert-base-uncased',
            'embedding_dim': 768,
            'description': 'BERT - 범용 모델'
        },
        'roberta': {
            'name': 'roberta-base',
            'embedding_dim': 768,
            'description': 'RoBERTa - 개선된 BERT'
        }
    }
    
    def __init__(self, price_data: pd.DataFrame, news_data: List[Dict], 
                 sequence_length: int = 60, prediction_length: int = 1,
                 device: Optional[torch.device] = None,
                 embedding_model: str = 'finbert'):
        """
        Args:
            price_data: 시계열 가격 데이터 (datetime, open, high, low, close, volume)
            news_data: 뉴스 데이터 리스트 [{timestamp, title, content, sentiment}, ...]
            sequence_length: 입력 시퀀스 길이 (과거 몇 개의 시간 단위를 볼지)
            prediction_length: 예측할 미래 길이
            device: GPU/CPU 장치 (None이면 자동 감지)
            embedding_model: 사용할 임베딩 모델 ('fine5', 'finbert', 'distilbert', 'bert', 'roberta')
        """
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.price_data = price_data.sort_values('datetime').reset_index(drop=True)
        self.news_data = sorted(news_data, key=lambda x: x['timestamp'])
        
        # 장치 설정
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # 가격 데이터 정규화
        self.price_mean = self.price_data[['open', 'high', 'low', 'close', 'volume']].mean()
        self.price_std = self.price_data[['open', 'high', 'low', 'close', 'volume']].std()
        self.normalized_prices = (self.price_data[['open', 'high', 'low', 'close', 'volume']] 
                                  - self.price_mean) / (self.price_std + 1e-8)
        
        # 임베딩 모델 설정
        embedding_model = embedding_model.lower()
        if embedding_model not in self.MODEL_CONFIGS:
            print(f"⚠ 경고: '{embedding_model}' 모델을 찾을 수 없습니다. 'finbert'를 사용합니다.")
            embedding_model = 'finbert'
        
        config = self.MODEL_CONFIGS[embedding_model]
        model_name = config['name']
        self.embedding_dim = config['embedding_dim']
        
        print(f"📝 임베딩 모델 로딩: {config['description']}")
        print(f"   모델: {model_name}")
        
        # 뉴스 임베딩을 위한 모델 로드
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Fin-E5는 Instruct 모델이지만 AutoModel로 임베딩 추출 가능
            self.news_model = AutoModel.from_pretrained(model_name)
            self.news_model.to(self.device)
            self.news_model.eval()
            print(f"✓ 모델 로드 완료 (임베딩 차원: {self.embedding_dim})")
        except Exception as e:
            print(f"⚠ 경고: {model_name} 로드 실패: {e}")
            print("   'bert-base-uncased'로 대체합니다.")
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.news_model = AutoModel.from_pretrained('bert-base-uncased')
            self.news_model.to(self.device)
            self.news_model.eval()
            self.embedding_dim = 768
        
        # 뉴스 데이터를 시간별로 그룹화
        self.news_by_time = self._group_news_by_time()
        
    def _group_news_by_time(self) -> Dict:
        """뉴스를 시간별로 그룹화"""
        news_dict = {}
        for news in self.news_data:
            time_key = pd.to_datetime(news['timestamp']).floor('H')  # 시간 단위로 그룹화
            if time_key not in news_dict:
                news_dict[time_key] = []
            news_dict[time_key].append(news)
        return news_dict
    
    def _get_news_embedding(self, news_list: List[Dict]) -> torch.Tensor:
        """뉴스 리스트의 평균 임베딩 계산"""
        if not news_list:
            return torch.zeros(self.embedding_dim, device=self.device)
        
        texts = [f"{n.get('title', '')} {n.get('content', '')[:200]}" for n in news_list]
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(text, return_tensors='pt', truncation=True, 
                                       max_length=128, padding='max_length')
                # 입력을 GPU로 이동
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.news_model(**inputs)
                # [CLS] 토큰의 임베딩 사용
                embeddings.append(outputs.last_hidden_state[0, 0, :])
        
        result = torch.stack(embeddings).mean(dim=0)
        # CPU로 이동 (DataLoader가 CPU에서 작동하므로)
        return result.cpu()
    
    def __len__(self):
        return len(self.price_data) - self.sequence_length - self.prediction_length + 1
    
    def __getitem__(self, idx):
        # 가격 시퀀스 추출
        price_seq = self.normalized_prices.iloc[idx:idx+self.sequence_length].values
        target_idx = idx + self.sequence_length
        target_price = self.normalized_prices.iloc[target_idx:target_idx+self.prediction_length]['close'].values
        
        # 해당 시간대의 뉴스 임베딩 추출
        time_key = pd.to_datetime(self.price_data.iloc[target_idx]['datetime']).floor('H')
        news_list = self.news_by_time.get(time_key, [])
        news_embedding = self._get_news_embedding(news_list)
        
        return {
            'price_sequence': torch.FloatTensor(price_seq),
            'news_embedding': news_embedding,
            'target': torch.FloatTensor(target_price)
        }


class PriceNewsGenerator(nn.Module):
    """가격과 뉴스를 통합하여 새로운 가격을 생성하는 모델"""
    
    def __init__(self, price_features: int = 5, news_embedding_dim: int = 768,
                 hidden_dim: int = 256, num_layers: int = 4, num_heads: int = 8):
        super(PriceNewsGenerator, self).__init__()
        
        # 가격 시퀀스 인코더 (Transformer)
        self.price_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # 가격 입력 프로젝션
        self.price_projection = nn.Linear(price_features, hidden_dim)
        
        # 뉴스 임베딩 프로젝션
        self.news_projection = nn.Linear(news_embedding_dim, hidden_dim)
        
        # 디코더 (가격 생성)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # 출력 레이어
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)  # 가격 예측
        )
        
        # 위치 인코딩
        self.pos_encoder = nn.Parameter(torch.randn(1, 1000, hidden_dim))
        
    def forward(self, price_sequence: torch.Tensor, news_embedding: torch.Tensor):
        """
        Args:
            price_sequence: [batch_size, seq_len, price_features]
            news_embedding: [batch_size, news_embedding_dim]
        """
        batch_size, seq_len, _ = price_sequence.shape
        
        # 가격 시퀀스 인코딩
        price_encoded = self.price_projection(price_sequence)  # [B, L, H]
        price_encoded = price_encoded + self.pos_encoder[:, :seq_len, :]
        encoded = self.price_encoder(price_encoded)
        
        # 뉴스 임베딩을 시퀀스 형태로 확장
        news_encoded = self.news_projection(news_embedding)  # [B, H]
        news_encoded = news_encoded.unsqueeze(1).expand(-1, seq_len, -1)  # [B, L, H]
        
        # 가격과 뉴스 정보 결합
        combined = encoded + news_encoded
        
        # 디코딩 (자기회귀적 생성)
        # 마지막 인코딩된 시퀀스를 디코더 입력으로 사용
        decoder_input = combined[:, -1:, :]  # [B, 1, H]
        outputs = []
        
        for _ in range(1):  # prediction_length만큼 생성
            decoded = self.decoder(decoder_input, combined)  # [B, 1, H]
            output = self.output_layer(decoded[:, -1:, :])  # [B, 1, 1]
            outputs.append(output)
            # 다음 반복을 위한 입력 (실제로는 range(1)이므로 사용되지 않지만 차원 맞춤)
            # decoder_input은 그대로 유지 (다음 예측 시 사용되지 않음)
        
        return torch.cat(outputs, dim=1)  # [B, 1, 1]


class StockPriceGenerator:
    """주식/코인 가격 생성기 메인 클래스"""
    
    def __init__(self, model_path: Optional[str] = None, embedding_model: str = 'finbert'):
        """
        Args:
            model_path: 저장된 모델 경로 (None이면 새로 생성)
            embedding_model: 사용할 임베딩 모델 ('fine5', 'finbert', 'distilbert', 'bert', 'roberta')
                             - 'fine5': 최고 성능 금융 임베딩 (16GB+ GPU 필요, Colab 권장)
                             - 'finbert': 금융 텍스트에 최적화 (권장, 기본값)
                             - 'distilbert': 더 빠르고 작은 모델
                             - 'bert': 범용 BERT 모델
                             - 'roberta': 개선된 BERT
        """
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.price_mean = None
        self.price_std = None
        self.embedding_model = embedding_model.lower()
        
        # GPU 정보 출력
        if torch.cuda.is_available():
            print(f"✓ GPU 감지됨: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA 버전: {torch.version.cuda}")
            print(f"  GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print("⚠ GPU를 사용할 수 없습니다. CPU로 실행합니다.")
            print("  더 빠른 학습을 위해 Google Colab 사용을 고려해보세요.")
        
        # 사용 가능한 모델 목록 출력
        if self.embedding_model in PriceNewsDataset.MODEL_CONFIGS:
            config = PriceNewsDataset.MODEL_CONFIGS[self.embedding_model]
            print(f"📝 임베딩 모델: {config['description']}")
        else:
            print(f"⚠ 경고: '{embedding_model}' 모델을 찾을 수 없습니다. 'finbert'를 사용합니다.")
            self.embedding_model = 'finbert'
        
        if model_path:
            self.load_model(model_path)
    
    def collect_price_data(self, symbol: str, period: str = '1y', interval: str = '1h') -> pd.DataFrame:
        """yfinance를 사용하여 가격 데이터 수집 (자동 fallback 포함)"""
        import sys
        from io import StringIO
        
        print(f"가격 데이터 수집 중: {symbol} (period={period}, interval={interval})")
        ticker = yf.Ticker(symbol)
        data = pd.DataFrame()
        
        # yfinance 경고를 임시로 억제하고 출력 캡처
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        # 시도할 조합 리스트 (우선순위 순서)
        attempts = [
            (period, interval),  # 원래 요청
        ]
        
        # fallback 조합 추가 - 항상 작동할 만한 조합들을 포함
        if interval == '1h':
            attempts.extend([
                ('1mo', '1d'),    # 가장 안정적인 조합 먼저
                ('3mo', '1d'),
                ('6mo', '1d'),
                (period, '1d'),   # 1h -> 1d로 변경
                ('6mo', interval), # period만 줄임
                ('3mo', interval),
                ('1mo', interval),
                (period, '1wk'),
            ])
        elif interval in ['5m', '15m', '30m', '60m']:
            attempts.extend([
                ('1mo', '1d'),    # 가장 안정적인 조합
                (period, '1h'),
                (period, '1d'),
                ('1mo', interval),
            ])
        else:
            attempts.extend([
                ('1mo', '1d'),    # 가장 안정적인 조합
                ('3mo', '1d'),
                ('6mo', '1d'),
                (period, '1d'),
                (period, '1wk'),
            ])
        
        # 각 조합 시도
        last_error = None
        for attempt_period, attempt_interval in attempts:
            try:
                sys.stdout = StringIO()  # 출력 초기화
                temp_data = ticker.history(period=attempt_period, interval=attempt_interval)
                output = sys.stdout.getvalue()
                sys.stdout = old_stdout
                
                # 경고 메시지 확인
                if 'No data found' in output or temp_data.empty or len(temp_data) == 0:
                    if attempt_period != period or attempt_interval != interval:
                        print(f"  [경고] {attempt_interval}/{attempt_period} 조합 실패, 다음 시도 중...")
                    continue
                
                # 성공한 경우
                data = temp_data
                if attempt_period != period or attempt_interval != interval:
                    print(f"  [성공] {attempt_interval}/{attempt_period} 조합으로 데이터 수집 성공!")
                break
                
            except Exception as e:
                sys.stdout = old_stdout
                last_error = str(e)
                if attempt_period != period or attempt_interval != interval:
                    print(f"  [경고] {attempt_interval}/{attempt_period} 조합 실패: {str(e)[:50]}")
                continue
        
        # stdout 복원
        if sys.stdout != old_stdout:
            sys.stdout = old_stdout
        
        if data.empty or len(data) == 0:
            raise ValueError(
                f"{symbol}: 데이터를 찾을 수 없습니다.\n\n"
                f"**가능한 원인:**\n"
                f"1. 심볼이 잘못되었거나 상장폐지되었을 수 있습니다\n"
                f"2. 선택한 기간({period})과 간격({interval}) 조합이 지원되지 않습니다\n"
                f"   - 1h 간격: 최대 60일 (권장: 1mo 기간)\n"
                f"   - 1d 간격: 제한 없음 (권장)\n"
                f"3. 인터넷 연결 또는 yahoo finance 서버 문제\n\n"
                f"**권장 설정:**\n"
                f"- 간격: 1d (일봉)\n"
                f"- 기간: 1y"
            )
        
        data.reset_index(inplace=True)
        # 컬럼명 정규화 (다양한 yfinance 버전 대응)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(0) if len(data.columns.levels) > 1 else data.columns
        
        data.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in data.columns]
        
        # 'adj_close' 컬럼이 없으면 'close'를 복사 (yfinance 버전별 차이 대응)
        if 'adj_close' not in data.columns and 'close' in data.columns:
            data['adj_close'] = data['close']
        
        # Datetime 컬럼 이름 처리 (다양한 경우 대응)
        if 'date' in data.columns:
            data.rename(columns={'date': 'datetime'}, inplace=True)
        elif 'index' in data.columns:
            data.rename(columns={'index': 'datetime'}, inplace=True)
        elif 'datetime' not in data.columns:
            # 인덱스가 이미 datetime이면 새로 생성
            if isinstance(data.index, pd.DatetimeIndex):
                data.insert(0, 'datetime', data.index)
            else:
                data.insert(0, 'datetime', pd.to_datetime(data.index))
        
        print(f"[완료] 데이터 수집 완료: {len(data)}개 레코드")
        return data
    
    def collect_news_data(self, symbol: str, days: int = 365, 
                         news_source: str = 'yfinance') -> List[Dict]:
        """
        뉴스 데이터 수집
        
        Args:
            symbol: 종목 심볼 (예: 'BTC-USD', 'AAPL')
            days: 수집할 일수
            news_source: 뉴스 소스 ('yfinance', 'newsapi', 'alphavantage', 'finnhub')
        
        Returns:
            뉴스 데이터 리스트 [{timestamp, title, content, sentiment}, ...]
        """
        print(f"뉴스 데이터 수집 중: {symbol} (소스: {news_source})")
        
        news_list = []
        
        try:
            if news_source == 'yfinance':
                news_list = self._fetch_news_yfinance(symbol, days)
            elif news_source == 'newsapi':
                news_list = self._fetch_news_newsapi(symbol, days)
            elif news_source == 'alphavantage':
                news_list = self._fetch_news_alphavantage(symbol, days)
            elif news_source == 'finnhub':
                news_list = self._fetch_news_finnhub(symbol, days)
            else:
                print(f"⚠ 경고: 알 수 없는 뉴스 소스 '{news_source}'. yfinance를 사용합니다.")
                news_list = self._fetch_news_yfinance(symbol, days)
        except Exception as e:
            print(f"⚠ 경고: {news_source}에서 뉴스 수집 실패: {e}")
            print("   yfinance로 재시도 중...")
            try:
                news_list = self._fetch_news_yfinance(symbol, days)
            except Exception as e2:
                print(f"⚠ yfinance도 실패: {e2}")
                news_list = []
        
        print(f"✓ 뉴스 데이터 수집 완료: {len(news_list)}개")
        return news_list
    
    def _fetch_news_yfinance(self, symbol: str, days: int) -> List[Dict]:
        """yfinance를 사용한 뉴스 수집 (무료, API 키 불필요)"""
        news_list = []
        
        try:
            ticker = yf.Ticker(symbol)
            news = getattr(ticker, 'news', None)
            if news is None:
                news = []
            if not isinstance(news, list):
                news = list(news) if news else []
            
            for item in news:
                try:
                    pt = item.get('providerPublishTime') or 0
                    if not pt or pt <= 0:
                        pt = int(datetime.now().timestamp())
                    timestamp = datetime.fromtimestamp(int(pt))
                    if (datetime.now() - timestamp).days > days:
                        continue
                    title = item.get('title', '') or item.get('headline', '')
                    if not title:
                        continue
                    news_list.append({
                        'timestamp': timestamp.isoformat(),
                        'title': title,
                        'content': item.get('summary', '') or item.get('description', '') or title,
                        'sentiment': 0,
                        'source': item.get('publisher', '') or item.get('source', 'Unknown'),
                        'url': item.get('link', '') or item.get('url', '')
                    })
                except Exception:
                    continue
        except Exception as e:
            print(f"  yfinance 뉴스 수집 오류: {e}")
        
        return news_list
    
    def _fetch_news_newsapi(self, symbol: str, days: int) -> List[Dict]:
        """NewsAPI를 사용한 뉴스 수집 (API 키 필요)"""
        news_list = []
        
        # API 키 확인 (환경 변수에서 가져오기 - 코드에 직접 적지 마세요!)
        api_key = (os.getenv('NEWSAPI_KEY', '') or '').strip()
        if not api_key:
            print("  ⚠ NEWSAPI_KEY 환경 변수가 설정되지 않았습니다.")
            print("     무료 API 키는 https://newsapi.org/register 에서 발급받을 수 있습니다.")
            return news_list
        
        try:
            # 심볼에서 종목명 추출 (예: BTC-USD -> Bitcoin)
            query = symbol.replace('-USD', '').replace('-', ' ')
            
            # NewsAPI 호출
            url = f"https://newsapi.org/v2/everything"
            params = {
                'q': f"{query} OR {symbol}",
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 100,
                'apiKey': api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') == 'ok':
                for article in data.get('articles', []):
                    try:
                        timestamp = datetime.fromisoformat(
                            article['publishedAt'].replace('Z', '+00:00')
                        )
                        
                        if (datetime.now(timestamp.tzinfo) - timestamp).days <= days:
                            news_list.append({
                                'timestamp': timestamp.isoformat(),
                                'title': article.get('title', ''),
                                'content': article.get('description', '') or article.get('title', ''),
                                'sentiment': 0,
                                'source': article.get('source', {}).get('name', 'Unknown'),
                                'url': article.get('url', '')
                            })
                    except Exception as e:
                        continue
        except Exception as e:
            print(f"  NewsAPI 오류: {e}")
        
        return news_list
    
    @staticmethod
    def _parse_av_time(s: str):
        """Alpha Vantage time_published 파싱. 예: 20240410T013000, 2024-04-10T01:30:00Z"""
        if not s or not isinstance(s, str):
            return None
        s = s.strip().replace('Z', '+00:00')
        try:
            # ISO: 2024-04-10T01:30:00 또는 2024-04-10T01:30:00+00:00
            return datetime.fromisoformat(s)
        except Exception:
            pass
        try:
            # compact: 20240410T013000 또는 20240410T013000-0500
            s0 = s[:15] if (len(s) >= 15 and s[8:9] == 'T') else s
            if len(s0) == 15 and s0[:8].isdigit() and s0[9:15].isdigit():
                return datetime(int(s0[0:4]), int(s0[4:6]), int(s0[6:8]), int(s0[9:11]), int(s0[11:13]), int(s0[13:15]))
        except Exception:
            pass
        return None
    
    def _fetch_news_alphavantage(self, symbol: str, days: int) -> List[Dict]:
        """Alpha Vantage NEWS_SENTIMENT API 사용 (API 키 필요)"""
        news_list = []
        
        api_key = os.getenv('ALPHAVANTAGE_API_KEY', '')
        if not api_key:
            print("  ⚠ ALPHAVANTAGE_API_KEY 환경 변수가 설정되지 않았습니다.")
            print("     무료 API 키는 https://www.alphavantage.co/support/#api-key 에서 발급받을 수 있습니다.")
            return news_list
        
        try:
            # 심볼에서 티커 추출 (예: BTC-USD -> BTC)
            ticker = symbol.split('-')[0]
            
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': ticker,
                'apikey': api_key,
                'limit': 1000
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'feed' in data:
                for item in data['feed']:
                    try:
                        ts = self._parse_av_time(item.get('time_published', ''))
                        if ts is None:
                            continue
                        timestamp = ts
                        
                        if (datetime.now() - timestamp).days <= days:
                            # 감정 점수 추출 (-1 to 1)
                            sentiment_score = 0
                            if 'overall_sentiment_score' in item:
                                try:
                                    sentiment_score = float(item['overall_sentiment_score'])
                                except:
                                    pass
                            
                            news_list.append({
                                'timestamp': timestamp.isoformat(),
                                'title': item.get('title', ''),
                                'content': item.get('summary', '') or item.get('title', ''),
                                'sentiment': sentiment_score,
                                'source': item.get('source', 'Unknown'),
                                'url': item.get('url', '')
                            })
                    except Exception as e:
                        continue
        except Exception as e:
            print(f"  Alpha Vantage 오류: {e}")
        
        return news_list
    
    def _fetch_news_finnhub(self, symbol: str, days: int) -> List[Dict]:
        """Finnhub API 사용 (API 키 필요)
        - 암호화폐(BTC-USD 등): /v1/news?category=crypto 사용 (company-news는 주식 전용)
        - 주식(AAPL 등): /v1/company-news 사용
        """
        news_list = []
        
        api_key = (os.getenv('FINNHUB_API_KEY', '') or '').strip()
        if not api_key:
            print("  ⚠ FINNHUB_API_KEY 환경 변수가 설정되지 않았습니다.")
            print("     무료 API 키는 https://finnhub.io/register 에서 발급받을 수 있습니다.")
            return news_list
        
        try:
            is_crypto = '-USD' in symbol.upper() or '-USDT' in symbol.upper()
            cutoff = (datetime.now() - timedelta(days=days)).timestamp()
            
            if is_crypto:
                # 암호화폐: company-news는 지원 안 함 → /v1/news?category=crypto
                url = "https://finnhub.io/api/v1/news"
                params = {'category': 'crypto', 'token': api_key}
            else:
                # 주식: /v1/company-news
                ticker = symbol.split('-')[0]
                url = "https://finnhub.io/api/v1/company-news"
                params = {
                    'symbol': ticker,
                    'from': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
                    'to': datetime.now().strftime('%Y-%m-%d'),
                    'token': api_key
                }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if isinstance(data, list):
                for item in data[:100]:
                    try:
                        ts = item.get('datetime', 0)
                        if ts and ts < cutoff:
                            continue
                        timestamp = datetime.fromtimestamp(ts) if ts else datetime.now()
                        news_list.append({
                            'timestamp': timestamp.isoformat(),
                            'title': item.get('headline', ''),
                            'content': item.get('summary', '') or item.get('headline', ''),
                            'sentiment': 0,
                            'source': item.get('source', 'Unknown'),
                            'url': item.get('url', '')
                        })
                    except Exception:
                        continue
        except Exception as e:
            print(f"  Finnhub 오류: {e}")
        
        return news_list
    
    def train(self, price_data: pd.DataFrame, news_data: List[Dict],
              epochs: int = 50, batch_size: int = 32, lr: float = 0.001):
        """모델 학습"""
        print(f"장치 정보: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU 사용 가능: {torch.cuda.get_device_name(0)}")
            print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print("GPU를 사용할 수 없습니다. CPU로 학습합니다.")
            print("더 빠른 학습을 위해 Google Colab 사용을 고려해보세요 (설정 방법은 README 참조)")
        
        print("데이터셋 생성 중...")
        dataset = PriceNewsDataset(price_data, news_data, device=self.device, 
                                   embedding_model=self.embedding_model)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                               num_workers=0 if self.device.type == 'cuda' else 2)
        
        # 정규화 파라미터 저장
        self.price_mean = dataset.price_mean
        self.price_std = dataset.price_std
        
        # 모델 초기화 (임베딩 차원을 데이터셋에서 가져옴)
        self.model = PriceNewsGenerator(
            news_embedding_dim=dataset.embedding_dim
        ).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        print(f"학습 시작 (장치: {self.device})...")
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                price_seq = batch['price_sequence'].to(self.device)
                news_emb = batch['news_embedding'].to(self.device)
                target = batch['target'].to(self.device)
                
                optimizer.zero_grad()
                output = self.model(price_seq, news_emb)  # [B, 1, 1]
                # target과 차원 맞추기: target은 [B, 1], output은 [B, 1, 1]
                target_reshaped = target.unsqueeze(-1) if target.dim() == 2 else target  # [B, 1, 1]
                loss = criterion(output, target_reshaped)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    def generate_price(self, price_history: pd.DataFrame, news_data: List[Dict],
                      steps: int = 10) -> np.ndarray:
        """새로운 가격 생성"""
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다. train()을 먼저 호출하세요.")
        
        self.model.eval()
        
        # 정규화
        normalized = (price_history[['open', 'high', 'low', 'close', 'volume']] 
                     - self.price_mean) / (self.price_std + 1e-8)
        
        generated_prices = []
        current_seq = torch.FloatTensor(normalized.values[-60:]).unsqueeze(0).to(self.device)
        
        # 뉴스 임베딩 계산
        dataset = PriceNewsDataset(price_history, news_data, device=self.device,
                                   embedding_model=self.embedding_model)
        news_emb = dataset._get_news_embedding(news_data).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            for _ in range(steps):
                output = self.model(current_seq, news_emb)
                pred_price = output[0, -1, 0].cpu().item()
                generated_prices.append(pred_price)
                
                # 시퀀스 업데이트 (새로운 예측값 추가)
                new_row = normalized.iloc[-1].copy()
                new_row['close'] = pred_price
                new_row_tensor = torch.FloatTensor(new_row.values).unsqueeze(0).unsqueeze(0)
                current_seq = torch.cat([current_seq[:, 1:, :], new_row_tensor], dim=1)
        
        # 역정규화
        generated_prices = np.array(generated_prices) * self.price_std['close'] + self.price_mean['close']
        
        return generated_prices
    
    def save_model(self, path: str):
        """모델 저장"""
        # 임베딩 차원 가져오기
        embedding_dim = self.model.news_projection.in_features if self.model else 768
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'price_mean': self.price_mean,
            'price_std': self.price_std,
            'embedding_model': self.embedding_model,
            'embedding_dim': embedding_dim
        }, path)
        print(f"모델이 저장되었습니다: {path}")
    
    def load_model(self, path: str):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # 저장된 임베딩 모델 정보 사용 (없으면 기본값)
        if 'embedding_model' in checkpoint:
            self.embedding_model = checkpoint['embedding_model']
        embedding_dim = checkpoint.get('embedding_dim', 768)
        
        self.model = PriceNewsGenerator(news_embedding_dim=embedding_dim).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.price_mean = checkpoint['price_mean']
        self.price_std = checkpoint['price_std']
        print(f"모델이 로드되었습니다: {path}")
        if 'embedding_model' in checkpoint:
            print(f"  임베딩 모델: {checkpoint['embedding_model']}")


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("주식/코인 가격 생성기 (LLM 기반)")
    print("=" * 60)
    
    # 생성기 초기화
    generator = StockPriceGenerator()
    
    # 예시: 비트코인 가격 데이터 수집
    print("\n1. 데이터 수집")
    price_data = generator.collect_price_data('BTC-USD', period='1y', interval='1h')
    print(f"수집된 가격 데이터: {len(price_data)}개")
    
    # 뉴스 데이터 수집 (실제로는 API를 통해 수집해야 함)
    news_data = generator.collect_news_data('BTC-USD', days=365)
    print(f"수집된 뉴스 데이터: {len(news_data)}개")
    
    # 모델 학습
    print("\n2. 모델 학습")
    if len(news_data) == 0:
        print("경고: 뉴스 데이터가 없습니다. 가격 데이터만으로 학습합니다.")
        # 뉴스 데이터가 없을 경우 빈 리스트로 진행
        news_data = [{'timestamp': price_data.iloc[i]['datetime'], 
                     'title': '', 'content': '', 'sentiment': 0} 
                    for i in range(len(price_data))]
    
    generator.train(price_data, news_data, epochs=50, batch_size=32)
    
    # 모델 저장
    print("\n3. 모델 저장")
    generator.save_model('price_generator_model.pt')
    
    # 가격 생성 예시
    print("\n4. 가격 생성 테스트")
    recent_prices = price_data.tail(100)
    generated = generator.generate_price(recent_prices, news_data[-10:], steps=10)
    print(f"생성된 가격 (다음 10단계):")
    for i, price in enumerate(generated):
        print(f"  Step {i+1}: ${price:.2f}")


if __name__ == "__main__":
    main()
