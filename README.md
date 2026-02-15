# 주식/코인 가격 생성기 (LLM 기반)

LLM과 Transformer를 활용하여 주식 및 암호화폐의 과거 가격 데이터와 뉴스 정보를 결합하여 새로운 가격을 생성하는 프로그램입니다.

## 주요 기능

- **다중 데이터 소스 통합**: 가격 시계열 데이터 + 뉴스 텍스트 데이터
- **Transformer 기반 모델**: 시계열 패턴과 뉴스 감성을 동시에 학습
- **자기회귀적 가격 생성**: LLM처럼 토큰을 생성하듯이 미래 가격을 단계적으로 생성
- **실시간 데이터 수집**: yfinance를 통한 가격 데이터 수집

## 설치

```bash
pip install -r requirements.txt
```

## 사용 방법

### 기본 사용

```python
from go_stock import StockPriceGenerator

# 생성기 초기화
generator = StockPriceGenerator()

# 데이터 수집
price_data = generator.collect_price_data('BTC-USD', period='1y', interval='1h')
news_data = generator.collect_news_data('BTC-USD', days=365)

# 모델 학습
generator.train(price_data, news_data, epochs=50)

# 가격 생성
future_prices = generator.generate_price(price_data.tail(100), news_data[-10:], steps=10)
```

### 실행

**GUI 실행 (권장):**
```bash
streamlit run gui.py
```

**커맨드라인 실행:**
```bash
python go_stock.py
```

## 모델 구조

1. **가격 인코더**: Transformer Encoder로 과거 가격 패턴 학습
2. **뉴스 임베딩**: BERT를 사용하여 뉴스 텍스트를 벡터로 변환
3. **통합 디코더**: 가격 패턴과 뉴스 정보를 결합하여 미래 가격 생성

## 연구 배경

이 프로그램은 다음과 같은 최신 연구들을 참고하여 구현되었습니다:

- **SDForger (IBM Research, 2025)**: LLM을 사용한 시계열 생성
- **Synergistic LLM-Transformer Architecture (2026)**: 뉴스 감성과 가격 예측 통합
- **StockMem (2025)**: 이벤트 기반 메모리 프레임워크

## 한계 및 주의사항

- **합성 데이터**: 생성된 가격은 실제 투자 결정에 사용하기 전에 충분한 검증이 필요합니다
- **뉴스 데이터**: 실제 뉴스 API 연동이 필요합니다 (현재는 예시 구조만 제공)
- **과거 편향**: 학습 데이터의 편향이 생성 결과에 반영될 수 있습니다

## 향후 개선 사항

- [ ] 실제 뉴스 API 연동 (Alpha Vantage, NewsAPI 등)
- [ ] 다중 종목 동시 학습 및 교차 상관관계 반영
- [ ] 변동성 클러스터링 등 금융 시계열 특성 보존
- [ ] 실시간 스트리밍 데이터 처리
- [x] 웹 인터페이스 추가 (Streamlit)

## 예측 정확도 개선 가이드 (실전 체크리스트)

질문하신 것처럼 **더 방대한 데이터 + 다양한 지표 + 생성 방식 점검**은 핵심이 맞습니다.
아래 순서로 진행하면 정확도를 실제로 개선하기 쉽습니다.

### 1) 데이터 품질/정합성 먼저 점검

- 가격/뉴스의 타임존을 UTC 기준으로 통일
- 뉴스 게시 시각과 캔들 시각 정렬(look-ahead leakage 방지)
- 결측치/중복/이상치(스파이크) 정제
- 거래소별 가격 차이(스프레드) 반영

### 2) 피처(지표) 확장

- 온체인 데이터: 거래소 순유입, 활성 주소, 펀딩비
- 파생시장 데이터: OI(미결제약정), 롱/숏 비율, basis
- 거시 변수: 금리, 달러지수(DXY), 위험자산 지수
- 뉴스 품질 개선: 출처 신뢰도 가중치, 이벤트 타입 분류(FOMC/ETF/해킹 등)

### 3) 검증 방식 강화 (가장 중요)

- 랜덤 분할 금지, **시계열 분할(walk-forward)** 사용
- MAE/RMSE 외에 방향성 정확도(Directional Accuracy) 추적
- 레짐(Bull/Bear/Volatile)별 성능을 분리 측정

### 4) 모델링 개선

- 학습 정규화 통계와 추론 정규화 통계를 일치
- 검증 손실 기준 Early Stopping 적용
- 단일 예측값보다 예측 구간(상/하단) 중심으로 의사결정

## 이번 코드에서 반영된 정확도 개선 포인트

- 학습/추론 정규화 통계 일치화 (데이터 누수 감소)
- 학습 구간/검증 구간 분리 + Early Stopping 추가
- 홀드아웃 구간 walk-forward 성능평가 메서드 추가 (`evaluate_walk_forward`)

예시:

```python
predictor = AIStockPredictor('BTC-USD')
predictor.collect_data(period='1y', interval='1h', news_days=30)
predictor.train_models(epochs=30, batch_size=32, lr=1e-3)

pred = predictor.predict()
metrics = predictor.evaluate_walk_forward(test_ratio=0.2)

print(pred.price, pred.confidence)
print(metrics)  # mae, rmse, mape, directional_accuracy
```

## 라이선스

MIT License
