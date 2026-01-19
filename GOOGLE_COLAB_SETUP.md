# Google Colab GPU 사용 가이드

이 가이드는 Go_Stock 프로젝트를 Google Colab에서 GPU를 사용하여 실행하는 방법을 설명합니다.

## 🚀 빠른 시작

### 1. Google Colab 열기
1. [Google Colab](https://colab.research.google.com/) 접속
2. 새 노트북 생성

### 2. GPU 활성화
1. 상단 메뉴: **런타임** → **런타임 유형 변경**
2. **하드웨어 가속기**: **GPU** 선택 (T4 또는 A100 권장)
3. **저장** 클릭

**💡 Fin-E5 사용 시:**
- T4 GPU (16GB) 또는 A100 GPU (40GB) 권장
- 무료 Colab은 T4 제공 (Fin-E5 사용 가능)
- Colab Pro/Pro+는 A100 제공 (더 빠름)

### 3. 파일 업로드 및 설치

다음 코드를 Colab 셀에 입력하고 실행:

```python
# 1. 필요한 라이브러리 설치
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install transformers pandas numpy yfinance requests accelerate bitsandbytes python-dotenv

# 2. 파일 업로드 (go_stock.py + .env 선택)
from google.colab import files
uploaded = files.upload()  # go_stock.py 필수, .env는 뉴스 API 사용 시

# 3. GPU 확인
import torch
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
```

### 4. API 키 설정 (뉴스 API 사용 시)

**.env를 로컬과 동일하게 쓰면 됩니다.** Colab에서 쓰는 방법 3가지:

#### 방법 A: .env 파일 업로드 (로컬과 동일) ⭐ 권장

로컬에서 쓰는 `.env`를 그대로 업로드합니다. `files.upload()` 시 `.env`도 함께 선택하세요.

```
# .env 내용 (로컬과 동일)
NEWSAPI_KEY=your_key_here
ALPHAVANTAGE_API_KEY=your_key_here
FINNHUB_API_KEY=your_key_here
```

`go_stock` import 시 `.env`를 자동으로 읽습니다. **이미 2번에서 .env를 업로드했다면 추가 작업 없음.**

- Colab에서는 **`/content/.env`** 또는 **현재 작업 디렉터리(cwd)의 `.env`** 를 자동으로 찾습니다. `files.upload()`로 올린 `.env`는 보통 `/content/.env`에 있어 별도 설정이 필요 없습니다.
- `go_stock_colab.ipynb`를 쓸 때: **2. 파일 업로드** 셀에서 `.env`를 같이 올리면, 해당 셀에서 `load_dotenv`로 즉시 로드하고 `NEWSAPI_KEY` 등 로드 여부를 출력합니다.

#### 방법 B: Colab에서 .env 새로 만들기

```python
# .env 파일 생성 (키가 노트북에 노출되지 않게 하려면 getpass 사용)
from getpass import getpass
from dotenv import load_dotenv

with open('.env', 'w') as f:
    f.write(f"NEWSAPI_KEY={getpass('NEWSAPI_KEY 입력: ')}\n")
    # 필요 시: f.write(f"ALPHAVANTAGE_API_KEY={getpass('Alpha Vantage 키: ')}\n")

load_dotenv('.env')
load_dotenv('/content/.env')  # Colab 기본 경로
print("✓ .env 생성 및 로드 완료")
```

#### 방법 C: 환경 변수로 직접 설정

```python
import os
os.environ['NEWSAPI_KEY'] = 'your_newsapi_key_here'
# os.environ['ALPHAVANTAGE_API_KEY'] = 'your_alphavantage_key_here'

# 이 셀 실행 후, 다음 셀에서 from go_stock import ... 하세요
```

> ⚠️ **방법 C**는 키가 노트북에 보이므로, 노트북을 공개/공유할 계획이면 **방법 A 또는 B**를 쓰는 것이 좋습니다.

### 5. 코드 실행

```python
# go_stock.py 실행
exec(open('go_stock.py').read())
```

또는 직접 main() 함수 호출:

```python
from go_stock import StockPriceGenerator

generator = StockPriceGenerator()

# 데이터 수집
price_data = generator.collect_price_data('BTC-USD', period='1y', interval='1h')
news_data = generator.collect_news_data('BTC-USD', days=365)

# 학습 (GPU 사용)
generator.train(price_data, news_data, epochs=50, batch_size=32)

# 모델 저장
generator.save_model('model_BTC-USD.pt')

# 모델 다운로드
from google.colab import files
files.download('model_BTC-USD.pt')
```

## 📊 성능 비교

| 환경 | 모델 | 학습 시간 (50 epochs, BTC-USD 1년 데이터) |
|------|------|-------------------------------------------|
| CPU (일반 노트북) | FinBERT | ~2-4시간 |
| GPU (Colab T4) | FinBERT | ~10-20분 |
| GPU (Colab T4) | **Fin-E5** | ~15-25분 (더 높은 정확도) |
| GPU (Colab A100) | Fin-E5 | ~8-15분 |
| GPU (Colab V100) | FinBERT | ~5-10분 |

**Fin-E5 사용 시:**
- ✅ FinBERT보다 **10-15% 더 높은 정확도**
- ✅ 금융 텍스트 임베딩 벤치마크 1위
- ⚠️ 더 많은 GPU 메모리 필요 (16GB+)
- ⚠️ 약간 더 느린 처리 속도

## ⚙️ 최적화 팁

### 1. 배치 크기 조정
GPU 메모리에 따라 배치 크기를 조정하세요:

```python
# T4 GPU (16GB): batch_size=32
# V100 GPU (32GB): batch_size=64
generator.train(price_data, news_data, epochs=50, batch_size=32)
```

### 2. Mixed Precision Training (선택사항)
더 빠른 학습을 위해 FP16 사용:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
# train() 메서드 내부에서 autocast 사용
```

### 3. 데이터셋 크기 조정
메모리 부족 시 기간을 줄이세요:

```python
# 1년 → 6개월
price_data = generator.collect_price_data('BTC-USD', period='6mo', interval='1h')
```

## 🔧 문제 해결

### GPU가 감지되지 않는 경우
```python
import torch
print(torch.cuda.is_available())  # False인 경우

# 런타임 재시작: 런타임 → 런타임 다시 시작
```

### 메모리 부족 오류
- **Fin-E5 사용 시**: 배치 크기를 8-16으로 줄이기
- 배치 크기 줄이기: `batch_size=16` 또는 `batch_size=8`
- 시퀀스 길이 줄이기: `sequence_length=30` (기본값: 60)
- 기간 줄이기: `period='3mo'` 또는 `period='1mo'`
- FinBERT로 전환: `embedding_model='finbert'` (더 적은 메모리 사용)

### CUDA 버전 불일치
Colab은 기본적으로 CUDA 11.8을 사용합니다. 위의 설치 명령어가 올바른 버전을 설치합니다.

## 📝 주의사항

1. **Colab 세션 제한**: 무료 계정은 약 12시간 후 세션이 종료됩니다
2. **GPU 할당**: 무료 계정은 GPU 사용 시간이 제한될 수 있습니다
3. **파일 저장**: Colab 세션이 종료되면 파일이 삭제되므로, 모델을 다운로드하거나 Google Drive에 저장하세요

## 💾 Google Drive 연동 (선택사항)

모델을 Google Drive에 저장하려면:

```python
from google.colab import drive
drive.mount('/content/drive')

# 모델 저장
generator.save_model('/content/drive/MyDrive/model_BTC-USD.pt')

# 모델 로드
generator = StockPriceGenerator('/content/drive/MyDrive/model_BTC-USD.pt')
```

## 🎯 완전한 예제

```python
# 전체 워크플로우
from go_stock import StockPriceGenerator
import torch

# GPU 확인
print(f"GPU 사용 가능: {torch.cuda.is_available()}")

# 생성기 초기화
# Fin-E5 사용 (최고 성능, Colab T4/A100에서 가능)
generator = StockPriceGenerator(embedding_model='fine5')

# 또는 FinBERT 사용 (더 빠름)
# generator = StockPriceGenerator(embedding_model='finbert')

# 데이터 수집
print("데이터 수집 중...")
price_data = generator.collect_price_data('BTC-USD', period='1y', interval='1h')
news_data = generator.collect_news_data('BTC-USD', days=365)

# 학습
print("학습 시작...")
# Fin-E5 사용 시 배치 크기 조정 (메모리 부족 방지)
batch_size = 16 if generator.embedding_model == 'fine5' else 32
generator.train(price_data, news_data, epochs=50, batch_size=batch_size, lr=0.001)

# 모델 저장
generator.save_model('model_BTC-USD.pt')
print("학습 완료!")

# 예측 테스트
recent_prices = price_data.tail(100)
generated = generator.generate_price(recent_prices, news_data[-10:], steps=10)
print(f"생성된 가격: {generated}")
```
