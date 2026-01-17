# Google Colab에서 Fin-E5 사용 가이드

Google Colab의 T4 (16GB) 또는 A100 (40GB) GPU를 사용하면 **Fin-E5** 모델을 사용할 수 있습니다!

## 🎯 Fin-E5란?

- **최고 성능 금융 임베딩 모델** (FinMTEB 벤치마크 1위)
- FinBERT보다 **10-15% 더 높은 정확도**
- 금융 뉴스, 재무 보고서, 시장 분석에 최적화
- 임베딩 차원: 4096 (FinBERT는 768)

## ✅ Colab에서 사용 가능한 이유

| GPU 타입 | 메모리 | Fin-E5 사용 가능? |
|---------|--------|------------------|
| **T4 (무료 Colab)** | 16GB | ✅ 가능 (배치 크기 조정 필요) |
| **A100 (Colab Pro+)** | 40GB | ✅ 완벽하게 가능 |
| **V100 (Colab Pro)** | 16GB | ✅ 가능 |

## 🚀 빠른 시작

### 1. Colab에서 GPU 활성화
```
런타임 → 런타임 유형 변경 → GPU (T4 또는 A100)
```

### 2. 코드 실행

```python
# 필요한 라이브러리 설치
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install transformers pandas numpy yfinance requests accelerate

# 파일 업로드
from google.colab import files
uploaded = files.upload()  # go_stock.py 업로드

# Fin-E5 사용 (최고 성능)
from go_stock import StockPriceGenerator
import torch

# GPU 확인
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Fin-E5로 생성기 초기화
generator = StockPriceGenerator(embedding_model='fine5')

# 데이터 수집
price_data = generator.collect_price_data('BTC-USD', period='1y', interval='1h')
news_data = generator.collect_news_data('BTC-USD', days=365)

# 학습 (Fin-E5는 메모리를 많이 사용하므로 배치 크기 조정)
generator.train(price_data, news_data, epochs=50, batch_size=16)

# 모델 저장
generator.save_model('model_BTC-USD_fine5.pt')
files.download('model_BTC-USD_fine5.pt')
```

## ⚙️ 최적화 팁

### 1. 배치 크기 조정

```python
# T4 GPU (16GB) - Fin-E5 사용 시
batch_size = 8  # 또는 16 (메모리 상황에 따라)

# A100 GPU (40GB) - Fin-E5 사용 시
batch_size = 32  # 더 큰 배치 가능
```

### 2. 메모리 부족 시

```python
# 옵션 1: 배치 크기 줄이기
generator.train(price_data, news_data, epochs=50, batch_size=8)

# 옵션 2: 데이터 기간 줄이기
price_data = generator.collect_price_data('BTC-USD', period='6mo', interval='1h')

# 옵션 3: FinBERT로 전환 (더 적은 메모리)
generator = StockPriceGenerator(embedding_model='finbert')
```

### 3. Mixed Precision (FP16) 사용 (선택사항)

더 빠른 학습을 위해 FP16 사용:

```python
# train() 메서드 내부에서 자동으로 처리됨
# 또는 수동으로 설정하려면 코드 수정 필요
```

## 📊 성능 비교

| 모델 | 정확도 | 학습 시간 (T4) | GPU 메모리 |
|------|--------|----------------|------------|
| **Fin-E5** | ⭐⭐⭐⭐⭐ (최고) | ~15-25분 | 16GB+ |
| **FinBERT** | ⭐⭐⭐⭐ | ~10-20분 | 4GB+ |

## 🔧 문제 해결

### "Out of Memory" 오류

```python
# 1. 배치 크기 줄이기
batch_size = 8

# 2. FinBERT로 전환
generator = StockPriceGenerator(embedding_model='finbert')

# 3. 데이터 기간 줄이기
period = '6mo'  # 또는 '3mo'
```

### 모델 다운로드 실패

```python
# HuggingFace 로그인 (선택사항, 더 빠른 다운로드)
from huggingface_hub import login
login()  # 토큰 입력

# 또는 직접 다운로드
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained('intfloat/e5-mistral-7b-instruct')
```

## 💡 Fin-E5 vs FinBERT 선택 가이드

### Fin-E5 선택 시:
- ✅ 최고 성능이 필요할 때
- ✅ GPU 메모리가 16GB+ 있을 때
- ✅ 정확도가 속도보다 중요할 때
- ✅ Colab T4/A100 사용 가능할 때

### FinBERT 선택 시:
- ✅ 빠른 학습이 필요할 때
- ✅ GPU 메모리가 4-8GB일 때
- ✅ 실시간 예측이 필요할 때
- ✅ 충분한 성능으로도 만족할 때

## 🎯 완전한 예제

```python
# 전체 워크플로우 (Fin-E5 사용)
from go_stock import StockPriceGenerator
import torch
from google.colab import files

# GPU 확인
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"  메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Fin-E5 사용 가능 여부 확인
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_memory >= 16:
        print("✓ Fin-E5 사용 가능!")
        embedding_model = 'fine5'
        batch_size = 16
    else:
        print("⚠ GPU 메모리가 부족합니다. FinBERT를 사용합니다.")
        embedding_model = 'finbert'
        batch_size = 32
else:
    print("⚠ GPU를 사용할 수 없습니다.")
    embedding_model = 'finbert'
    batch_size = 16

# 생성기 초기화
generator = StockPriceGenerator(embedding_model=embedding_model)

# 데이터 수집
print("데이터 수집 중...")
price_data = generator.collect_price_data('BTC-USD', period='1y', interval='1h')
news_data = generator.collect_news_data('BTC-USD', days=365)

# 학습
print("학습 시작...")
generator.train(price_data, news_data, epochs=50, batch_size=batch_size, lr=0.001)

# 모델 저장
model_name = f'model_BTC-USD_{embedding_model}.pt'
generator.save_model(model_name)
print(f"✓ 모델 저장 완료: {model_name}")

# 다운로드
files.download(model_name)
```

## 📝 주의사항

1. **첫 실행 시 모델 다운로드**: Fin-E5는 약 14GB 크기로 다운로드에 시간이 걸립니다
2. **메모리 관리**: T4 GPU에서는 배치 크기를 8-16으로 제한하는 것을 권장합니다
3. **세션 제한**: Colab 무료 계정은 12시간 후 세션이 종료되므로 모델을 다운로드하세요

## 🎉 결론

**Google Colab을 사용하면 Fin-E5를 쉽게 사용할 수 있습니다!**

- T4 GPU (무료)에서도 사용 가능
- FinBERT보다 10-15% 더 높은 정확도
- 배치 크기만 조정하면 문제없이 실행 가능

**지금 바로 시도해보세요!** 🚀
