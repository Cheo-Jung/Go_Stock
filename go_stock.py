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
import warnings
warnings.filterwarnings('ignore')


class PriceNewsDataset(Dataset):
    """가격과 뉴스 데이터를 결합한 데이터셋"""
    
    def __init__(self, price_data: pd.DataFrame, news_data: List[Dict], 
                 sequence_length: int = 60, prediction_length: int = 1):
        """
        Args:
            price_data: 시계열 가격 데이터 (datetime, open, high, low, close, volume)
            news_data: 뉴스 데이터 리스트 [{timestamp, title, content, sentiment}, ...]
            sequence_length: 입력 시퀀스 길이 (과거 몇 개의 시간 단위를 볼지)
            prediction_length: 예측할 미래 길이
        """
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.price_data = price_data.sort_values('datetime').reset_index(drop=True)
        self.news_data = sorted(news_data, key=lambda x: x['timestamp'])
        
        # 가격 데이터 정규화
        self.price_mean = self.price_data[['open', 'high', 'low', 'close', 'volume']].mean()
        self.price_std = self.price_data[['open', 'high', 'low', 'close', 'volume']].std()
        self.normalized_prices = (self.price_data[['open', 'high', 'low', 'close', 'volume']] 
                                  - self.price_mean) / (self.price_std + 1e-8)
        
        # 뉴스 임베딩을 위한 토크나이저 (FinBERT 또는 일반 BERT 사용)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.news_model = AutoModel.from_pretrained('bert-base-uncased')
        self.news_model.eval()
        
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
            return torch.zeros(768)  # BERT base의 hidden size
        
        texts = [f"{n.get('title', '')} {n.get('content', '')[:200]}" for n in news_list]
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(text, return_tensors='pt', truncation=True, 
                                       max_length=128, padding='max_length')
                outputs = self.news_model(**inputs)
                # [CLS] 토큰의 임베딩 사용
                embeddings.append(outputs.last_hidden_state[0, 0, :])
        
        return torch.stack(embeddings).mean(dim=0)
    
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
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.price_mean = None
        self.price_std = None
        
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
    
    def collect_news_data(self, symbol: str, days: int = 365) -> List[Dict]:
        """뉴스 데이터 수집 (예시: Alpha Vantage API 또는 다른 소스 사용)"""
        print(f"뉴스 데이터 수집 중: {symbol}")
        # 실제 구현에서는 뉴스 API를 사용해야 함
        # 여기서는 예시 데이터 구조만 제공
        news_list = []
        
        # 예시: 실제로는 API 호출
        # news_list = self._fetch_news_from_api(symbol, days)
        
        return news_list
    
    def train(self, price_data: pd.DataFrame, news_data: List[Dict],
              epochs: int = 50, batch_size: int = 32, lr: float = 0.001):
        """모델 학습"""
        print("데이터셋 생성 중...")
        dataset = PriceNewsDataset(price_data, news_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 정규화 파라미터 저장
        self.price_mean = dataset.price_mean
        self.price_std = dataset.price_std
        
        # 모델 초기화
        self.model = PriceNewsGenerator().to(self.device)
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
        dataset = PriceNewsDataset(price_history, news_data)
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
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'price_mean': self.price_mean,
            'price_std': self.price_std
        }, path)
        print(f"모델이 저장되었습니다: {path}")
    
    def load_model(self, path: str):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model = PriceNewsGenerator().to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.price_mean = checkpoint['price_mean']
        self.price_std = checkpoint['price_std']
        print(f"모델이 로드되었습니다: {path}")


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
