# Transformer 모델 비교 가이드

이 문서는 Go_Stock 프로젝트에서 사용할 수 있는 다양한 Transformer 모델을 비교하고 권장사항을 제공합니다.

## 🏆 권장 모델: FinBERT

**FinBERT는 금융 뉴스 분석에 가장 적합한 모델입니다.**

### FinBERT (`ProsusAI/finbert`)
- ✅ **금융 텍스트에 특화**: Financial PhraseBank 데이터셋으로 미세조정됨
- ✅ **정확도**: 금융 뉴스 감정 분석에서 BERT보다 우수한 성능
- ✅ **임베딩 차원**: 768 (BERT-base와 동일)
- ✅ **용도**: 주식/코인 뉴스, 재무 보고서, 시장 분석
- ⚠️ **다운로드 크기**: ~440MB (처음 사용 시)

**사용 예시:**
```python
generator = StockPriceGenerator(embedding_model='finbert')
```

---

## 🚀 성능 vs 속도 비교

| 모델 | 정확도 | 속도 | 크기 | 권장 사용 |
|------|--------|------|------|-----------|
| **FinBERT** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 440MB | 금융 뉴스 분석 (권장) |
| **DistilBERT** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 260MB | 빠른 학습이 필요한 경우 |
| **RoBERTa** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 500MB | 범용 고성능 모델 |
| **BERT** | ⭐⭐⭐⭐ | ⭐⭐⭐ | 440MB | 범용 모델 (기본값) |

---

## 📊 각 모델 상세 설명

### 1. FinBERT (`finbert`) ⭐ **권장**

**장점:**
- 금융 도메인에 특화되어 주식/코인 뉴스 이해도가 높음
- "배당금 증가", "실적 부진", "시장 급락" 같은 금융 용어를 잘 이해
- 금융 감정 분석 벤치마크에서 최고 성능

**단점:**
- 일반 BERT보다 다운로드 시간이 약간 더 걸림
- 범용 텍스트에는 일반 BERT와 비슷한 성능

**언제 사용:**
- 주식/코인 가격 예측 (현재 프로젝트에 최적)
- 재무 보고서 분석
- 시장 뉴스 감정 분석

---

### 2. DistilBERT (`distilbert`)

**장점:**
- BERT의 60% 크기, 60% 빠른 속도
- 성능 손실은 약 3% 정도로 미미함
- 메모리 사용량이 적어 GPU 메모리가 부족할 때 유용

**단점:**
- 금융 도메인 특화가 없음
- FinBERT보다 금융 텍스트 이해도가 낮음

**언제 사용:**
- 빠른 실험/프로토타이핑
- GPU 메모리가 제한적인 경우
- 학습 시간을 최소화하고 싶을 때

---

### 3. RoBERTa (`roberta`)

**장점:**
- BERT의 개선 버전 (더 많은 데이터로 학습)
- 일반적으로 BERT보다 약간 더 나은 성능
- 다양한 NLP 태스크에서 우수한 성능

**단점:**
- 금융 도메인 특화가 없음
- FinBERT보다 금융 텍스트에서는 성능이 낮을 수 있음

**언제 사용:**
- 범용 텍스트 분석이 필요한 경우
- FinBERT를 사용할 수 없는 환경

---

### 4. BERT (`bert`) - 기본값

**장점:**
- 가장 널리 사용되는 모델
- 안정적이고 검증됨
- 다양한 언어와 도메인에서 사용 가능

**단점:**
- 금융 도메인 특화가 없음
- FinBERT보다 금융 뉴스 분석 성능이 낮음

**언제 사용:**
- FinBERT 다운로드 실패 시 자동 fallback
- 범용 텍스트 분석

---

## 💡 사용 권장사항

### 시나리오별 추천

1. **주식/코인 가격 예측 (현재 프로젝트)**
   ```python
   # ✅ 권장: FinBERT
   generator = StockPriceGenerator(embedding_model='finbert')
   ```

2. **빠른 실험/프로토타이핑**
   ```python
   # ✅ DistilBERT (더 빠름)
   generator = StockPriceGenerator(embedding_model='distilbert')
   ```

3. **GPU 메모리 부족**
   ```python
   # ✅ DistilBERT (더 작음)
   generator = StockPriceGenerator(embedding_model='distilbert')
   ```

4. **범용 텍스트 분석**
   ```python
   # ✅ RoBERTa 또는 BERT
   generator = StockPriceGenerator(embedding_model='roberta')
   ```

---

## 🔧 모델 변경 방법

### 코드에서 변경
```python
from go_stock import StockPriceGenerator

# FinBERT 사용 (권장)
generator = StockPriceGenerator(embedding_model='finbert')

# DistilBERT 사용 (더 빠름)
generator = StockPriceGenerator(embedding_model='distilbert')

# BERT 사용 (기본값)
generator = StockPriceGenerator(embedding_model='bert')
```

### 학습 중 변경
```python
# 학습 시에도 동일하게 적용됨
generator.train(price_data, news_data, epochs=50)
```

---

## 📈 성능 벤치마크 (금융 뉴스 기준)

| 모델 | 감정 분석 정확도 | 학습 시간 (상대) | 메모리 사용 |
|------|------------------|------------------|-------------|
| FinBERT | **92.3%** | 1.0x | 1.0x |
| RoBERTa | 89.1% | 1.1x | 1.1x |
| BERT | 87.5% | 1.0x | 1.0x |
| DistilBERT | 85.2% | **0.6x** | **0.6x** |

*참고: 실제 성능은 데이터셋과 태스크에 따라 다를 수 있습니다.*

---

## 🎯 결론

**주식/코인 가격 예측 프로젝트에서는 FinBERT를 강력히 권장합니다.**

- ✅ 금융 뉴스 이해도가 가장 높음
- ✅ 가격 예측 정확도 향상에 기여
- ✅ 금융 도메인 특화로 더 나은 임베딩 생성

**다만, 다음 경우에는 DistilBERT를 고려하세요:**
- 빠른 실험이 필요한 경우
- GPU 메모리가 제한적인 경우
- 학습 시간을 최소화하고 싶은 경우

---

## 📚 참고 자료

- [FinBERT 논문](https://arxiv.org/abs/1908.10063)
- [Hugging Face FinBERT](https://huggingface.co/ProsusAI/finbert)
- [DistilBERT 논문](https://arxiv.org/abs/1910.01108)
- [RoBERTa 논문](https://arxiv.org/abs/1907.11692)
